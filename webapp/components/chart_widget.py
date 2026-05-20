import streamlit as st
import io


def show_chart(fig, caption: str = None, use_container_width: bool = True):
    """Hiển thị matplotlib figure với tùy chọn download."""
    st.pyplot(fig, use_container_width=use_container_width)
    if caption:
        st.caption(caption)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    label = caption or "chart"
    filename = label.replace(" ", "_").replace("/", "_")[:40] + ".png"
    st.download_button(
        label="Tải PNG",
        data=buf,
        file_name=filename,
        mime="image/png",
        key=f"dl_{filename}_{id(fig)}",
    )
