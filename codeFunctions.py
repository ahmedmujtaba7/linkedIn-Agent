import re
import cloudinary
import requests
import os
from pathlib import Path

from config import ImageType
from imageFunctions import generate_linkedin_image


def code_to_image(
    code: str,
    language: str = "javascript",
    theme: str = "one-hunter",
    mode: str = "dark",
    background: str = "midnight",
    padding: int = 0,
    font_size: int = 14,
    line_numbers: bool = True,
    title: str = "",
    scale: int = 1,
) -> bytes:
    """
    Convert code to an image using Ray.so API.
    
    Args:
        code: The code to convert to image
        language: Programming language (default: "javascript")
        theme: Color theme (default: "one-hunter")
        mode: Display mode (default: "dark")
        background: Background style (default: "midnight")
        padding: Padding around code (default: 32)
        font_size: Font size (default: 14)
        line_numbers: Show line numbers (default: True)
        title: Title to display (default: "")
        scale: Image scale/quality (default: 2)
    
    Returns:
        PNG image as bytes
    """
    payload = {
        "code": code,
        "language": language,
        "theme": theme,
        "mode": mode,
        "background": background,
        "padding": padding,
        "fontSize": font_size,
        "lineNumbers": line_numbers,
        "title": title,
        "scale": scale,
    }
    
    response = requests.post(
        "https://ray.tinte.dev/api/v1/screenshot",
        json=payload,
        timeout=30,
    )
    
    if not response.ok:
        raise Exception(f"Ray API error {response.status_code}: {response.text}")
    
    return response.content  # PNG image bytes


def save_code_image(
    code: str,
    output_dir: str = "./generated_images",
    filename: str = "code_snippet.png",
    **kwargs,
) -> str:
    """
    Convert code to image and save locally.
    
    Args:
        code: The code to convert
        output_dir: Directory to save image (default: "./generated_images")
        filename: Output filename (default: "code_snippet.png")
        **kwargs: Additional options for code_to_image
    
    Returns:
        Path to saved image file
    """
    image_bytes = code_to_image(code, **kwargs)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = out_dir / filename
    
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    
    print(f"[✓] Code image saved → {save_path.resolve()}")
    return str(save_path.resolve())


    
 
def upload_to_cloudinary(image_path: str) -> str:
    """Uploads image to Cloudinary and returns public URL"""

    print("☁️ Uploading image to Cloudinary...")

    result = cloudinary.uploader.upload(
        image_path,
        folder="linkedin_bot",
        resource_type="image",
    )

    image_url = result["secure_url"]
    print(f"🌍 Cloudinary URL → {image_url}")
    return image_url





def generate_code_snippet_image(topic: str, code: str, language: str = "Python", **kwargs) -> str:
    ext = {"python": "py", "javascript": "js", "typescript": "ts", "html": "html"}.get(
        language.lower(), language.lower()[:3] or "txt"
    )
    filename = kwargs.pop("filename", f"snippet.{ext}")
    return generate_linkedin_image(
        ImageType.CODE_SNIPPET,
        topic=topic,
        content=code,
        language=language,
        filename=filename,
        **kwargs,
    )


def extract_code_from_post(post: str) -> tuple[str | None, str]:
    """
    Extract code block and language from LinkedIn post.
    
    Returns:
        (code, language) or (None, "") if no code found
    """
    import re
    
    # Match markdown code blocks: ```language\ncode\n```
    pattern = r'```(\w+)?\s*\n(.*?)\n```'
    match = re.search(pattern, post, re.DOTALL)
    
    if match:
        language = match.group(1) or "python"
        code = match.group(2).strip()
        return code, language
    
    return None, ""


def replace_code_with_image_url(post: str, image_url: str) -> tuple[str, str]:
    """
    Replace the code block in the post with a placeholder.
    
    Args:
        post: The original post content
        image_url: URL of the code image to insert
    
    Returns:
        Tuple of (updated_post_text, image_url)
        The updated_post_text has the code block removed,
        and image_url is returned separately for LinkedIn media attachment.
    """
    # Remove the code block entirely from the post text
    pattern = r'```\w+\s*\n.*?\n```'
    updated_post = re.sub(pattern, "", post, flags=re.DOTALL).strip()
    
    # Add a subtle reference to code at the end before hashtags
    # Find where hashtags start
    hashtag_pattern = r'\n(#\w+)'
    hashtag_match = re.search(hashtag_pattern, updated_post)
    
    if hashtag_match:
        # Insert before hashtags
        insertion_point = hashtag_match.start()
        updated_post = updated_post[:insertion_point] + "\n\n👇 Check out the code snippet in the image below!\n" + updated_post[insertion_point:]
    else:
        # No hashtags, just append at the end
        updated_post += "\n\n👇 Check out the code snippet in the image below!"
    
    return updated_post, image_url


def generate_code_image_from_post(post: str, groq_client) -> dict | None:
    """
    Extract code from post, generate image using Ray.so API,
    upload to Cloudinary, and return image URL.
    
    Returns:
        {"image_url": str, "code": str, "language": str} or None
    """
    code, language = extract_code_from_post(post)
    
    if not code:
        print("ℹ️  No code snippet found in post.")
        return None
    
    try:
        print(f"🎨 Generating code image for {language}...")
        
        # Save code image locally
        filename = f"code_{language.lower()}_snippet.png"
        local_path = save_code_image(
            code,
            language=language.lower(),
            title=language,
            filename=filename,
        )
        
        # Upload to Cloudinary
        print("☁️ Uploading code image to Cloudinary...")
        image_url = upload_to_cloudinary(local_path)
        
        # Clean up local file
        try:
            os.remove(local_path)
            print(f"🗑️  Local file deleted → {local_path}")
        except OSError as e:
            print(f"⚠️  Could not delete local file: {e}")
         
        return {
            "image_url": image_url,
            "code": code,
            "language": language,
        }
    
    except Exception as e:
        print(f"❌ Error generating code image: {e}")
        return None




