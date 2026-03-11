# Image Filter Studio

A polished Streamlit application for applying image filters with OpenCV.

## Features

- Upload PNG/JPG images
- Built-in sample images for quick testing
- Multiple filters: Black & White, Brightness, Style, Vintage, HDR
- Adjustable filter parameters
- Download processed output as PNG
- Input validation and user-friendly error handling

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Project Structure

```text
pro2/
  app.py
  requirements.txt
  README.md
```

## Notes

- For smoother performance, use images under 2MB.
- The app is intentionally self-contained and does not require external assets.
