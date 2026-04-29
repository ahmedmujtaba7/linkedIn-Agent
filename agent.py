import os
import calendar
import re
import requests
from datetime import datetime

import cloudinary
import cloudinary.uploader
from groq import Groq


from config import GEMINI_API_KEY, GROQ_API_KEY, LINKEDIN_ACCESS_TOKEN, ImageType
from imageFunctions import generate_image_from_spec, get_image_decision
from codeFunctions import generate_code_image_from_post, replace_code_with_image_url, upload_to_cloudinary


import cloudinary

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
)

# ─── DAY THEMES (No Sunday) ───────────────────────────
DAY_THEMES = {
    "Monday": "motivational story or personal experience related to coding, office life, career, or tech journey",
    "Tuesday": "technical explanation of a backend concept in NestJS or Node.js with practical examples",
    "Wednesday": "software architecture or system design concept explained clearly with real world use cases",
    "Thursday": "frontend React/Next JS development topic, or web security and performance optimization technique with practicle examples",
    "Friday": "advanced database concept, query optimization, or data modeling technique",
    "Saturday": "building with AI, creating agents, automation, or practical LLM use cases for developers",
}


def get_todays_theme():
    day = calendar.day_name[datetime.now().weekday()]
    if day not in DAY_THEMES:
        print(f"📅 Today is {day} — No post scheduled for Sunday. Exiting.")
        return None, None
    theme = DAY_THEMES[day]
    print(f"📅 Today is {day}")
    print(f"🎯 Theme: {theme}")
    return day, theme

# ─── GENERATE POST ────────────────────────────────────
def generate_post():
    day, theme = get_todays_theme()

    if not day:
        return None

    groq_client = Groq(api_key=GROQ_API_KEY)

    is_technical = day not in ["Monday"]
    dm_instruction = (
        "- End with this line before hashtags: '💬 Have questions or working on something similar? DM me — happy to help.'"
        if is_technical
        else ""
    )
    tone_instruction = (
        "Share a personal, vulnerable, real story or experience. Be human and inspiring."
        if not is_technical
        else "Explain clearly with a practical example or analogy. Teach something useful."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a senior software engineer and LinkedIn content creator.
                You write posts that feel genuinely human — never robotic, never generic.
                Every post you write must be on a DIFFERENT and FRESH topic.
                Never repeat the same idea twice. Be creative and specific.""",
                            },
                            {
                                "role": "user",
                                "content": f"""Today is {day}.

                                Your job:
                                1. First, PICK a fresh and specific topic related to: {theme}
                                2. Then write a LinkedIn post about that topic

                                Post requirements:
                                - Start with a curiosity-driven hook to get readers attention (no emojis in first line)
                                - Dont start techincal posts with "In today's world", "As a developer", "In this post", "what If questions"
                                - {tone_instruction}
                                - Conversational tone — write like a real person, not a textbook
                                - Include a practical example or code snippet if relevant to the theme. Use triple backticks to format the code block and specify the language, for example, javascript, python, or html.
                                - When adding code snippet add complete code snippet and exclude dependencies.
                                - If including a code snippet, make it at least 15 and maximum 20 lines long — short snippets look bad as images
                                - Under 400 words
                                - Do not discuss outdated concepts or technologies.
                                - Always Generate Unique content, never copy from other sources.
                                - Always discuss Upto Date topics and technologies. for example in ReactJs concepts like class based components are outdated and now we use functional components with hooks so dont discuss such outdated topics.
                                - Use short paragraphs and line breaks for readability
                                - End with a question that invites comments
                                {dm_instruction}
                                - Add 8-10 relevant hashtags at the very bottom
                                - NEVER start with clichés like 'In today's world', 'As a developer', 'In this post'
                                - NEVER write the same topic twice — be specific and original every time""",
                            },
                ],
                )

    post = response.choices[0].message.content
    print("✅ Post generated!\n")
    print("─" * 50)
    print(post)
    print("─" * 50)

    image_spec = get_image_decision(post, groq_client)
    if image_spec is None:
        return post

    return [post, image_spec]                


def get_profile_urn(token):
    resp = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch profile URN: {resp.status_code} {resp.text}")
    data = resp.json()
    urn = f"urn:li:person:{data['sub']}"
    print(f"👤 Profile URN: {urn}")
    return urn


def post_to_linkedin(content, image_url=None):
    token = LINKEDIN_ACCESS_TOKEN

    print("🔐 Authenticating with LinkedIn API...")
    urn = get_profile_urn(token)

    if image_url:
        # Step 1: Register the image with LinkedIn to get an image URN
        print("📸 Registering image with LinkedIn...")
        register_response = requests.post(
            "https://api.linkedin.com/v2/assets?action=registerUpload",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            },
            json={
                "registerUploadRequest": {
                    "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                    "owner": urn,
                    "serviceRelationships": [
                        {
                            "relationshipType": "OWNER",
                            "identifier": "urn:li:userGeneratedContent",
                        }
                    ],
                }
            },
        )
        
        if register_response.status_code != 200:
            print(f"⚠️  Failed to register image: {register_response.status_code}")
            print(f"Response: {register_response.text}")
            # Fall back to text-only post
            image_url = None
        else:
            register_data = register_response.json()
            upload_url = register_data.get("value", {}).get("uploadMechanism", {}).get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {}).get("uploadUrl")
            asset_urn = register_data.get("value", {}).get("asset")
            
            if not upload_url or not asset_urn:
                print("⚠️  Could not extract upload URL or asset URN")
                image_url = None
            else:
                # Step 2: Download image from Cloudinary
                print(f"⬇️  Downloading image from {image_url[:50]}...")
                img_response = requests.get(image_url)
                if img_response.status_code != 200:
                    print(f"⚠️  Failed to download image: {img_response.status_code}")
                    image_url = None
                else:
                    # Step 3: Upload to LinkedIn
                    print("⬆️  Uploading image to LinkedIn...")
                    upload_response = requests.put(
                        upload_url,
                        data=img_response.content,
                        headers={"Content-Type": "image/png"},
                    )
                    
                    if upload_response.status_code != 201:
                        print(f"⚠️  Failed to upload image: {upload_response.status_code}")
                        image_url = None
                    else:
                        image_url = asset_urn

    # Create payload based on whether we have an image
    if image_url and isinstance(image_url, str) and image_url.startswith("urn:"):
        # Image was successfully uploaded
        payload = {
            "author": urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": content,
                    },
                    "shareMediaCategory": "IMAGE",
                    "media": [
                        {
                            "status": "READY",
                            "media": image_url,
                        }
                    ],
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
            },
        }
    else:
        # Text-only post
        payload = {
            "author": urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": content,
                    },
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
            },
        }

    print("📤 Publishing post via LinkedIn API...")
    resp = requests.post(
        "https://api.linkedin.com/v2/ugcPosts",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        },
        json=payload,
    )

    if resp.status_code == 201:
        post_id = resp.headers.get("x-restli-id", "unknown")
        print("✅ Successfully posted to LinkedIn!")
        print(f"🔗 Post ID: {post_id}")
    else:
        print(f"❌ Failed to post: {resp.status_code}")
        print(f"📋 Response: {resp.text}")
        raise Exception(f"LinkedIn API error: {resp.status_code}")

if __name__ == "__main__":
    print(f"🤖 LinkedIn Agent started at {datetime.now()}")

    content = generate_post()
    GeneratePostImageFlag = "false"  # For Gemini post images
    GenerateCodeImageFlag = "true"   # For Ray.so code images
    PostToLinkedinFlag = "true"

    if content is None:
        print("😴 No post today — enjoy your Sunday!")

    elif isinstance(content, list):
        post_text, image_spec = content[0], content[1]

        if(GeneratePostImageFlag == "true"):

            if not GEMINI_API_KEY:
                print("⚠️  GEMINI_API_KEY / IMAGE_MODEL_API_KEY not set — cannot generate image.")
            else:
                local_path = generate_image_from_spec(image_spec)
                cloudinary_url = upload_to_cloudinary(local_path)
                try:
                    os.remove(local_path)
                    print(f"🗑️  Local file deleted → {local_path}")
                except OSError as e:
                    print(f"⚠️  Could not delete local file: {e}")

                print(f"\n🌍 Post image live at: {cloudinary_url}")
        
        # Generate code image if enabled (independent from post image)
        if GenerateCodeImageFlag == "true":
            groq_client = Groq(api_key=GROQ_API_KEY)
            code_image_result = generate_code_image_from_post(post_text, groq_client)
            
            code_image_url = None
            if code_image_result:
                print(f"[OK] Code image generated and uploaded!")
                print(f"[OK] Image URL: {code_image_result['image_url']}")
                print(f"[OK] Code language: {code_image_result['language']}")
                
                # Replace code block with image URL in the post
                code_image_url = code_image_result['image_url']
                post_text, _ = replace_code_with_image_url(post_text, code_image_url)
                print(f"\n📋 Updated post content (code replaced with image):\n{post_text}\n")

        if(PostToLinkedinFlag == "true"):
            post_to_linkedin(post_text, image_url=code_image_url if code_image_url else None)

    else:
        
        post_text = content
        
        # Generate code image if enabled (independent from post image)
        code_image_url = None
        if GenerateCodeImageFlag == "true":
            groq_client = Groq(api_key=GROQ_API_KEY)
            code_image_result = generate_code_image_from_post(post_text, groq_client)
            
            if code_image_result:
                print(f"[OK] Code image generated and uploaded!")
                print(f"[OK] Image URL: {code_image_result['image_url']}")
                print(f"[OK] Code language: {code_image_result['language']}")
                
                # Replace code block with image URL in the post
                code_image_url = code_image_result['image_url']
                post_text, _ = replace_code_with_image_url(post_text, code_image_url)
                print(f"\n📋 Updated post content (code replaced with image):\n{post_text}\n")
        
        if(PostToLinkedinFlag == "true"):
            post_to_linkedin(post_text, image_url=code_image_url)

    print(f"\n🏁 Agent finished at {datetime.now()}")
