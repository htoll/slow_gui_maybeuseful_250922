# read_movie.py
# SIF Movie Exporter (MP4/MOV/TIFF) — Region, Labels, Bottom Colorbar, Flip-X, Compact Preview

from __future__ import annotations

import os
import gc
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize, LogNorm

# I/O backends for video/TIFF
try:
    import imageio
except Exception as e:
    imageio = None
    _imageio_err = e

# Prefer imageio-ffmpeg for MP4/MOV
try:
    import imageio_ffmpeg  # noqa: F401
    _has_ffmpeg = True
except Exception:
    _has_ffmpeg = False

try:
    import tifffile
except Exception as e:
    tifffile = None
    _tif_err = e

# PIL for text/legend overlays & resizing
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    Image = None
    ImageDraw = None
    ImageFont = None
    _pil_err = e

# SIF reader
try:
    import sif_parser  # must provide np_open(path, ignore_corrupt=True) -> (frames[T,H,W], meta)
except Exception as e:
    sif_parser = None
    _sif_import_error = e

OVERLAY_SCALE = 2  # supersampling for higher res text


@dataclass
class Meta:
    exposure_ms: Optional[float]
    gain: Optional[float]


def _np_open_all(path: str) -> Tuple[np.ndarray, Dict]:
    if sif_parser is None:
        raise RuntimeError(f"`sif_parser` not importable: {_sif_import_error}")
    frames, meta = sif_parser.np_open(path, ignore_corrupt=True)
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None, ...]
    return frames, meta


def _to_cps(frame2d: np.ndarray, meta: Dict) -> Tuple[np.ndarray, Meta]:
    gain_dac = meta.get("GainDAC", 1) or 1
    exposure_time = meta.get("ExposureTime", 1.0) or 1.0  # seconds
    acc = meta.get("AccumulatedCycles", 1) or 1
    cps = frame2d * (5.0 / gain_dac) / exposure_time / acc
    return cps, Meta(exposure_ms=float(exposure_time) * 1000.0, gain=float(gain_dac))


def _crop_region(arr: np.ndarray, region: str) -> np.ndarray:
    region = str(region)
    h, w = arr.shape[-2], arr.shape[-1]
    mid_h, mid_w = h // 2, w // 2
    if region == '3':
        return arr[0:mid_h, 0:mid_w]
    elif region == '4':
        return arr[0:mid_h, mid_w:w]
    elif region == '1':
        return arr[mid_h:h, 0:mid_w]
    elif region == '2':
        return arr[mid_h:h, mid_w:w]
    elif region == 'custom':
        y0, y1, x0, x1 = 312, min(512, h), 56, min(256, w)
        return arr[y0:y1, x0:x1]
    return arr


def _compute_norm(selected_idxs: List[int], frames: np.ndarray, meta: Dict, log_scale: bool, region: str) -> Optional[Normalize]:
    vmin = np.inf
    vmax = -np.inf
    for i in selected_idxs:
        cps, _ = _to_cps(frames[i], meta)
        cps = _crop_region(cps, region)
        fi_min = float(np.nanmin(cps))
        fi_max = float(np.nanmax(cps))
        if fi_min < vmin:
            vmin = fi_min
        if fi_max > vmax:
            vmax = fi_max
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return LogNorm() if log_scale else None
    return LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)


def _cps_to_uint8(cps: np.ndarray, norm: Optional[Normalize]) -> np.ndarray:
    if norm is None:
        mn = float(np.nanmin(cps))
        mx = float(np.nanmax(cps))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(cps, dtype=np.uint8)
        scaled = (cps - mn) / (mx - mn)
    else:
        scaled = norm(cps)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * 255.0).astype(np.uint8)


def _apply_colormap(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    if cmap_name.lower() in ("gray", "grey"):
        return gray_u8  # single-channel grayscale
    cmap = mpl_cm.get_cmap(cmap_name)
    rgba = cmap(gray_u8.astype(np.float32) / 255.0)  # (H,W,4) float
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)   # (H,W,3)
    return rgb




def _overlay_labels(frame_rgb_or_gray: np.ndarray, text: str) -> np.ndarray:
    if Image is None:
        return frame_rgb_or_gray
    if frame_rgb_or_gray.ndim == 2:
        base = np.stack([frame_rgb_or_gray]*3, axis=-1)
    else:
        base = frame_rgb_or_gray
    H = base.shape[0]
    font_px = max(16, int(0.04 * H))  # ~4% of height
    font = _get_font(font_px)

    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    margin = max(4, font_px // 4)
    try:
        x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = x2 - x1, y2 - y1
    except Exception:
        text_w, text_h = font_px * 8, font_px
    pad = max(4, font_px // 4)
    box = [margin, margin, margin + text_w + 2*pad, margin + text_h + 2*pad]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((margin + pad, margin + pad), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def _make_colorbar_with_ticks(
    height: int,
    cmap_name: str,
    vmin_val: float,
    vmax_val: float,
    log_scale: bool,
) -> np.ndarray:
    """
    Build a vertical color bar with 5 ticks, scientific-notation labels, and a 'cps' units label.
    Returns an RGB np.ndarray of shape (H, strip_w + panel_w, 3).
    """
    H = int(max(1, height))
    strip_w = 28
    panel_w = 110  # room for 5 labels like 1.23e+04
    bar_h = int(round(0.75 * H))
    top_pad = (H - bar_h) // 2

    # ---- Build the color strip (bar_h x strip_w x 3) ----
    grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]  # 1 at top -> 0 at bottom

    def _strip_rgb():
        name = (cmap_name or "").lower()
        if name in ("gray", "grey"):
            strip = (grad * 255.0).astype(np.uint8)                   # (bar_h, 1)
            strip = np.repeat(strip, strip_w, axis=1)                 # (bar_h, strip_w)
            return np.stack([strip, strip, strip], axis=-1)           # (bar_h, strip_w, 3)
        else:
            try:
                cmap = mpl_cm.get_cmap(cmap_name)
            except Exception:
                cmap = mpl_cm.get_cmap("viridis")
            rgba = cmap(grad)                                         # (bar_h, 1, 4)
            rgb = (rgba[..., :3] * 255.0).astype(np.uint8)            # (bar_h, 1, 3)
            return np.repeat(rgb, strip_w, axis=1)                    # (bar_h, strip_w, 3)

    strip_rgb = _strip_rgb()

    # ---- Create final canvas (one place to draw/paste everything) ----
    canvas = Image.new("RGB", (strip_w + panel_w, H), (0, 0, 0))
    WHITE = (255, 255, 255)

    # Paste the strip centered vertically with top_pad
    canvas.paste(Image.fromarray(strip_rgb), (0, top_pad))
    draw_canvas = ImageDraw.Draw(canvas)

    # ---- Tick values + normalization ----
    finite = np.isfinite(vmin_val) and np.isfinite(vmax_val)
    use_log = log_scale and finite and (vmin_val > 0) and (vmax_val > vmin_val)

    if use_log:
        ticks_vals = np.geomspace(vmin_val, vmax_val, 5)
        den = (np.log(vmax_val) - np.log(vmin_val))
        def norm_fn(v: float) -> float:
            return (np.log(v) - np.log(vmin_val)) / den
    else:
        if not finite or vmax_val == vmin_val:
            # fall back to a simple 0..1 range to avoid division by zero/NaN
            vmin_val, vmax_val = 0.0, 1.0
        ticks_vals = np.linspace(vmin_val, vmax_val, 5)
        den = (vmax_val - vmin_val)
        def norm_fn(v: float) -> float:
            return (v - vmin_val) / den

    def val_to_y(v: float) -> int:
        """Map value v into canvas Y (0 at top)."""
        t = float(np.clip(norm_fn(float(v)), 0.0, 1.0))
        y_bar = int(round((1.0 - t) * (bar_h - 1)))   # 0..bar_h-1, 0 is top of bar
        return top_pad + y_bar

    # ---- Draw tick marks on the strip edge (on the same canvas) ----
    tick_len = 6
    x_edge = strip_w - 1
    for v in ticks_vals:
        y = val_to_y(v)
        draw_canvas.line([(x_edge - tick_len, y), (x_edge, y)], fill=WHITE, width=1)

    # ---- Supersampled label panel ----
    # Use configured OVERLAY_SCALE if present; otherwise default to 2x
    scale = globals().get("OVERLAY_SCALE", 2)
    panel_hi = Image.new("RGB", (panel_w * scale, H * scale), (0, 0, 0))
    draw_panel = ImageDraw.Draw(panel_hi)

    # Font
    font_px = max(12, int(0.032 * H)) * scale
    try:
        font = _get_font(font_px)  # use your existing font loader
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()

    # Tick labels
    label_pad_x = 6 * scale
    for v in ticks_vals:
        y_hi = val_to_y(v) * scale
        label = f"{v:.2e}"
        try:
            bbox = draw_panel.textbbox((0, 0), label, font=font)
            _, _, tw, th = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 60 * scale, 12 * scale
        draw_panel.text((label_pad_x, max(0, y_hi - th // 2)), label, fill=WHITE, font=font)

    # Units (top-right, away from ticks)
    units = "pps"
    try:
        bbox_u = draw_panel.textbbox((0, 0), units, font=font)
        uw, uh = bbox_u[2] - bbox_u[0], bbox_u[3] - bbox_u[1]
    except Exception:
        uw, uh = 28 * scale, 12 * scale
    unit_x = panel_w * scale - uw - 6 * scale
    unit_y = max(6 * scale, top_pad * scale - uh - 6 * scale)
    draw_panel.text((unit_x, unit_y), units, fill=WHITE, font=font)

    # Downsample the label panel and paste onto the same final canvas
    panel_lo = panel_hi.resize((panel_w, H), resample=Image.LANCZOS)
    canvas.paste(panel_lo, (strip_w, 0))

    # ---- Return the finished image ----
    return np.array(canvas)




def _concat_right(img: np.ndarray, right: Optional[np.ndarray]) -> np.ndarray:
    if right is None:
        return img
    H = img.shape[0]
    if right.shape[0] != H:
        if Image is not None:
            pil = Image.fromarray(right)
            pil = pil.resize((right.shape[1], H), resample=Image.NEAREST)
            right = np.array(pil)
        else:
            right = np.resize(right, (H, right.shape[1], right.shape[-1]))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if right.ndim == 2:
        right = np.stack([right]*3, axis=-1)
    return np.concatenate([img, right], axis=1)

def _concat_bottom(img: np.ndarray, bottom: Optional[np.ndarray]) -> np.ndarray:
    if bottom is None:
        return img
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if bottom.ndim == 2:
        bottom = np.stack([bottom]*3, axis=-1)
    H, W = img.shape[:2]
    if bottom.shape[1] != W:
        if Image is not None:
            pil = Image.fromarray(bottom)
            pil = pil.resize((W, bottom.shape[0]), resample=Image.NEAREST)
            bottom = np.array(pil)
        else:
            bottom = np.resize(bottom, (bottom.shape[0], W, bottom.shape[-1]))
    return np.concatenate([img, bottom], axis=0)


def _encode_video(frames_u8: List[np.ndarray], fps: int, format_ext: str) -> bytes:
    if imageio is None:
        raise RuntimeError(f"imageio not available: {_imageio_err}")
    if not _has_ffmpeg:
        raise RuntimeError("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")
    safe_frames = []
    for f in frames_u8:
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=-1)
        safe_frames.append(f)
    if format_ext.lower() == "mp4":
        codec = "libx264"; pixelformat = "yuv420p"
    elif format_ext.lower() == "mov":
        codec = "libx264"; pixelformat = "yuv420p"
    else:
        raise ValueError("Unsupported video format for encoder")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_ext}") as tmp:
        tmp_path = tmp.name
    try:
        writer = imageio.get_writer(tmp_path, fps=fps, codec=codec, pixelformat=pixelformat, quality=8, format="FFMPEG")
        for f in safe_frames:
            writer.append_data(f)
        writer.close()
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _encode_tiff_stack(frames_u8: List[np.ndarray]) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
        tmp_path = tmp.name
    try:
        if tifffile is not None:
            arr = np.stack(frames_u8, axis=0)
            tifffile.imwrite(tmp_path, arr)
        else:
            if imageio is None:
                raise RuntimeError(f"Neither tifffile nor imageio available (tifffile error: {_tif_err}, imageio error: {_imageio_err})")
            imageio.mimwrite(tmp_path, frames_u8, format="TIFF")
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _get_font(size: int):
    if ImageFont is None:
        return None
    for name in [
        "Arial.ttf", "arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def run():
    st.title("SIF Movie Exporter")


    with st.sidebar:
        st.header("Controls")
        colormap = st.selectbox("Colormap", ["gray", "magma", "viridis", "plasma", "hot", "hsv", "cividis", "inferno"], index=0)
        univ_min_max = st.checkbox("Universal min/max across frames", value=False)
        log_scale = st.checkbox("Log intensity scaling", value=False)
        region = st.selectbox("Region", options=["all", "1", "2", "3", "4"], index=0)
        show_colorbar = st.checkbox("Show colorbar", value=True)
        show_labels = st.checkbox("Show frame # / exposure / gain", value=True)
        fps = st.slider("FPS (preview & video)", 1, 60, 15)
        export_fmt = st.selectbox("Download format", ["MP4", "MOV", "TIFF"], index=0)

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)
    if not uploaded:
        st.info("Upload a .sif file to begin.")
        return
    uploaded_name = uploaded.name  # e.g. "experiment1.sif"
    base = os.path.splitext(os.path.basename(uploaded_name))[0]
    today = date.today().strftime("%Y%m%d")

    raw_frames = None
    frames_u8: List[np.ndarray] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
        tmp.write(uploaded.getbuffer())
        sif_path = tmp.name

    try:
        raw_frames, meta_raw = _np_open_all(sif_path)
        T = int(raw_frames.shape[0])
        if T == 0:
            st.error("No frames found in SIF file.")
            return

        idxs = list(range(0, T, 1))  # show all frames
        norm = _compute_norm(idxs, raw_frames, meta_raw, log_scale, region) if univ_min_max else None

        # Build full-res frames (with flip, labels, bottom colorbar)
        for i in idxs:
            cps, md = _to_cps(raw_frames[i], meta_raw)
            cps = _crop_region(cps, region)

            cps = np.flip(cps, axis=0) #flip x axis
            u8 = _cps_to_uint8(cps, norm)
            rgb_or_gray = _apply_colormap(u8, colormap)

            strip = None
            if show_colorbar:
                if norm is not None:
                    vmin_val = float(getattr(norm, 'vmin', np.nan))
                    vmax_val = float(getattr(norm, 'vmax', np.nan))
                else:
                    vmin_val = float(np.nanmin(cps))
                    vmax_val = float(np.nanmax(cps))
                strip = _make_colorbar_with_ticks(
                    height=rgb_or_gray.shape[0],
                    cmap_name=colormap,
                    vmin_val=vmin_val,
                    vmax_val=vmax_val,
                    log_scale=log_scale,
                )
            framed = _concat_right(rgb_or_gray, strip)


            if show_labels:
                if (md.exposure_ms is not None) and (md.gain is not None):
                    label = f"Frame {i+1} | Exp {md.exposure_ms:g} ms | Gain {md.gain:g}"
                else:
                    label = f"Frame {i+1}"
                framed = _overlay_labels(framed, label)

            frames_u8.append(framed)

        # Build smaller frames for preview
        frames_preview = []
        for f in frames_u8:
            if Image is not None:
                pil = Image.fromarray(f if f.ndim == 3 else np.stack([f]*3, axis=-1))
                w, h = pil.size
                frames_preview.append(np.array(pil))
            else:
                frames_preview.append(f)

        # Preview
        col_preview, col_empty = st.columns([1, 2])
        with col_preview:
            st.subheader("Preview")
            if _has_ffmpeg:
                try:
                    preview_bytes = _encode_video(frames_u8, fps=fps, format_ext="mp4")
                    st.video(preview_bytes)
                except Exception as e:
                    st.warning(f"MP4 preview unavailable: {e}")
                    if imageio is not None:
                        from io import BytesIO
                        gif_buf = BytesIO()
                        imageio.mimsave(gif_buf, frames_u8, format="GIF", duration=1.0 / max(1, fps))
                        st.image(gif_buf.getvalue(), caption="GIF preview (encoding fallback)", output_format="GIF")
                    else:
                        st.info("Install `imageio` for GIF preview.")
            else:
                if imageio is not None:
                    from io import BytesIO
                    gif_buf = BytesIO()
                    imageio.mimsave(gif_buf, frames_u8, format="GIF", duration=1.0 / max(1, fps))
                    st.image(gif_buf.getvalue(), caption="GIF preview (FFmpeg not available)", output_format="GIF")
                else:
                    st.info("Install `imageio` for GIF preview.")


        # Downloads
        try:
            if export_fmt in ("MP4", "MOV"):
                if not _has_ffmpeg:
                    st.error("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")
                    # also offer TIFF immediately
                    dl_bytes = _encode_tiff_stack(frames_u8)
                    today = date.today().strftime("%Y%m%d")
                    st.download_button(
                        label="Download TIFF",
                        data=dl_bytes,
                        file_name=f"{base}_{today}.tiff",
                        mime="image/tiff",
                    )
                else:
                    ext = export_fmt.lower()
                    dl_bytes = _encode_video(frames_u8, fps=fps, format_ext=ext)
                    mime = "video/mp4" if ext == "mp4" else "video/quicktime"
                    today = date.today().strftime("%Y%m%d")
                    st.download_button(
                        label=f"Download {export_fmt}",
                        data=dl_bytes,
                        file_name=f"{base}_{today}.{ext}",
                        mime=mime,
                    )
            else:
                dl_bytes = _encode_tiff_stack(frames_u8)
                today = date.today().strftime("%Y%m%d")
                st.download_button(
                    label="Download TIFF",
                    data=dl_bytes,
                    file_name=f"{base}_{today}.tiff",
                    mime="image/tiff",
                )
        except Exception as e:
            st.error(f"Export failed: {e}")

    finally:
        # Safe cleanup (no UnboundLocalError)
        try:
            if raw_frames is not None:
                del raw_frames
        except Exception:
            pass
        try:
            frames_u8  # ensure defined
            del frames_u8
        except Exception:
            pass
        gc.collect()
