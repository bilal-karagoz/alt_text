import streamlit as st
import pandas as pd
from PIL import Image
import google.generativeai as genai
import deepl
import os
import io
import json # Retained for potential future use, though not for API keys
import base64
import requests

# --- Page Configuration ---
st.set_page_config(page_title="AI Alt Text Generator & Translator", layout="wide") # Changed title

# --- Static Configuration ---
GOOGLE_AI_MODEL_NAME = 'gemini-2.0-flash' # Consistent model name, or gemini-2.0-flash as user had
DEFAULT_PROMPT = "Describe this image concisely for use as website alt text. Focus on the main subject, any important context and the marketing intent. Avoid phrases like 'Image of' or 'Picture of'. If the image contains clearly legible and important text, incorporate that text naturally into the description if it's central to understanding the image."
DEFAULT_MAX_TOKENS = 150
THUMBNAIL_SIZE = (500, 500)


DEEPL_EUROPEAN_LANGUAGES = {
    "Croatian": "HR", "Czech": "CS", "Danish": "DA", "Dutch": "NL",
    "Estonian": "ET", "Finnish": "FI", "French": "FR",
    "German": "DE", "Greek": "EL", "Hebrew": "HE", "Hungarian": "HU",
    "Italian": "IT", "Latvian": "LV", "Lithuanian": "LT", "Norwegian": "NB",
    "Polish": "PL", "Portuguese": "PT-PT", "Romanian": "RO", "Russian": "RU",
    "Slovak": "SK", "Slovenian": "SL", "Spanish": "ES", "Swedish": "SV",
    "Turkish": "TR", "Ukrainian": "UK",
}

# --- Helper Functions ---
try:
    from google.generativeai.types import FinishReason
    USE_FINISH_REASON_ENUM = True
except ImportError:
    USE_FINISH_REASON_ENUM = False
    MANUAL_FINISH_REASON_MAP = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}

def get_thumbnail_base64(image_bytes, size=THUMBNAIL_SIZE):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail(size)
        buffered = io.BytesIO()
        img_format = img.format if img.format else 'PNG'

        if img_format.upper() == 'JPEG':
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
        elif img_format.upper() == 'PNG':
            pass
        else:
            if img.mode == 'P':
                 img = img.convert('RGBA')
            elif img.mode == 'RGBA' and img_format.upper() != 'PNG':
                 img = img.convert('RGB')

        img.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{img_format.lower()};base64,{img_str}"
    except Exception as e:
        return None

def generate_alt_text_google(image_bytes, custom_prompt=None, custom_max_tokens=None):
    global GOOGLE_AI_MODEL_NAME
    try:
        model = genai.GenerativeModel(GOOGLE_AI_MODEL_NAME)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        prompt_to_use = custom_prompt if custom_prompt and custom_prompt.strip() else DEFAULT_PROMPT
        max_tokens_to_use = custom_max_tokens if custom_max_tokens is not None else DEFAULT_MAX_TOKENS

        generation_config = {"temperature": 0.4, "top_p": 1, "top_k": 32, "max_output_tokens": max_tokens_to_use}
        safety_settings=[
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in
            ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
             "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        response = model.generate_content([prompt_to_use, img], generation_config=generation_config, safety_settings=safety_settings)

        if response.text: return response.text.strip()
        if response.candidates and response.candidates[0]:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                if text_parts: return " ".join(text_parts).strip()
            reason = candidate.finish_reason
            name = f"RAW_{reason}"
            if USE_FINISH_REASON_ENUM:
                try: name = FinishReason(reason).name
                except ValueError: name = f"UNMAPPED_{reason}"
            elif reason in MANUAL_FINISH_REASON_MAP: name = MANUAL_FINISH_REASON_MAP[reason]
            return f"Error: Text gen failed (Reason: {name})"
        return "Error: No text/candidate info."
    except Exception as e: return f"Error: Google AI call failed ({type(e).__name__}. Check API Key & model or image format.)" # Added type(e)

def translate_text_deepl(api_key, text, target_lang, source_lang="EN"):
    if not text or (isinstance(text, str) and text.startswith("Error:")): return text
    if not api_key: return "Error: DeepL API Key not provided."
    try:
        translator = deepl.Translator(api_key)
        return translator.translate_text(text, source_lang=source_lang, target_lang=target_lang).text
    except Exception as e: return f"Error: DeepL ({target_lang}): {type(e).__name__}"

def to_excel(df):
    df_for_excel = df.copy()
    cols_to_drop = ["Preview", "ImageIndex"]
    for col in cols_to_drop:
        if col in df_for_excel.columns:
            df_for_excel = df_for_excel.drop(columns=[col])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df_for_excel.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def update_alt_text_in_df(df_session_key, image_index_value, widget_key):
    if widget_key in st.session_state and df_session_key in st.session_state:
        edited_text = st.session_state[widget_key]
        df = st.session_state[df_session_key]
        target_df_indices = df[df['ImageIndex'] == image_index_value].index
        if not target_df_indices.empty:
            df_index = target_df_indices[0]
            df.loc[df_index, 'English Alt Text'] = edited_text
            st.session_state[df_session_key] = df

def handle_google_api_key_change():
    if "google_api_input_widget" in st.session_state:
        st.session_state.google_api_key = st.session_state.google_api_input_widget
        if st.session_state.google_api_key:
            try:
                genai.configure(api_key=st.session_state.google_api_key)
                st.session_state.google_api_configured_status = True
            except Exception as e:
                st.session_state.google_api_configured_status = f"Configuration Error: {type(e).__name__}"
        else:
            st.session_state.google_api_configured_status = None

def handle_deepl_api_key_change():
    if "deepl_api_input_widget" in st.session_state:
        st.session_state.deepl_api_key = st.session_state.deepl_api_input_widget

if 'app_initialized' not in st.session_state:
    st.session_state.google_api_key = ""
    st.session_state.deepl_api_key = ""
    st.session_state.google_api_configured_status = None
    st.session_state.custom_prompt = DEFAULT_PROMPT
    st.session_state.custom_max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.generated_results = None
    st.session_state.translated_results = None
    st.session_state.image_input_method = "Upload Files"
    
    if st.session_state.google_api_key: # Should not run initially as key is empty
        try:
            genai.configure(api_key=st.session_state.google_api_key)
            st.session_state.google_api_configured_status = True
        except Exception as e:
            st.session_state.google_api_configured_status = f"Auto-config Error: {type(e).__name__}"
    st.session_state.app_initialized = True

# --- Streamlit App UI ---
st.title("🖼️ AI Alt Text Generator & Translator") # Title changed

with st.sidebar.expander("🔑 API Keys Configuration", expanded=not bool(st.session_state.get('google_api_key'))):
    st.markdown("Enter your API keys. Keys are used for the current session only.")
    st.text_input("Google AI API Key", type="password", key="google_api_input_widget",
                  on_change=handle_google_api_key_change,
                  value=st.session_state.get('google_api_key', ""), 
                  help="Required for generating alt text.")
    if st.session_state.get('google_api_configured_status') is True: 
        st.caption("✔️ Google AI Key active for this session.")
    elif isinstance(st.session_state.get('google_api_configured_status'), str): 
        st.error(st.session_state.google_api_configured_status)
    elif not st.session_state.get('google_api_key'): 
        st.caption("Google AI Key not yet entered for this session.")

    st.text_input("DeepL API Key", type="password", help="Optional. Only needed for translation features.",
                  key="deepl_api_input_widget",
                  on_change=handle_deepl_api_key_change,
                  value=st.session_state.get('deepl_api_key', ""))
    if st.session_state.get('deepl_api_key'): 
        st.caption("✔️ DeepL Key entered for this session.")
    else: 
        st.caption("DeepL Key not entered (translation disabled for this session).")

st.sidebar.header("⚙️ Operation Mode")
app_mode = st.sidebar.radio("Choose what you want to do:",
                            ("Generate Alt Text from Images", "Translate Existing Alt Text from Excel File"),
                            key="app_mode_selection")

if app_mode == "Generate Alt Text from Images":
    st.header("📸 Generate Alt Text from Images")
    st.markdown(f"Upload images or provide URLs, and the AI (using `{GOOGLE_AI_MODEL_NAME}`) will generate English alt text.")

    st.session_state.image_input_method = st.radio(
        "Choose image input method:",
        ("Upload Files", "Enter Image URLs"),
        key="image_input_method_radio",
        horizontal=True,
        index=["Upload Files", "Enter Image URLs"].index(st.session_state.get("image_input_method", "Upload Files"))
    )

    if st.session_state.image_input_method == "Upload Files":
        uploaded_files_list = st.file_uploader("Upload JPEG, JPG, or PNG images:", type=["jpeg", "jpg", "png"],
                                          accept_multiple_files=True, key="image_uploader_gen_mode")
    else:
        image_urls_input_str = st.text_area(
            "Enter image URLs (one per line):",
            height=150,
            key="image_urls_input_area",
            placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.png"
        )

    with st.expander("⚙️ Advanced Prompt Settings (Optional)", expanded=False):
        st.markdown("Customize the prompt for alt text generation and control max output tokens.")
        custom_prompt_input = st.text_area("Custom Prompt:", value=st.session_state.get('custom_prompt', DEFAULT_PROMPT),
                                           height=150, key="custom_prompt_input_widget",
                                           help="Customize the AI's instructions. Leave as default if unsure.") # Help text updated
        st.session_state.custom_prompt = custom_prompt_input
        custom_max_tokens_input = st.number_input("Max Output Tokens:", min_value=50, max_value=1024,
                                                  value=st.session_state.get('custom_max_tokens', DEFAULT_MAX_TOKENS),
                                                  step=10, key="custom_max_tokens_input_widget",
                                                  help="Max length of generated alt text. Default: 150.")
        st.session_state.custom_max_tokens = custom_max_tokens_input

    if st.button("✨ Generate English Alt Text", type="primary", key="generate_button_main"):
        if st.session_state.get('google_api_configured_status') is not True:
            st.error("❌ Google AI API Key is not configured correctly or is missing. Please check the sidebar.")
        else:
            image_sources_to_process = []
            source_type = ""
            if st.session_state.image_input_method == "Upload Files":
                if not uploaded_files_list:
                    st.error("❌ Please upload at least one image."); st.stop()
                image_sources_to_process = uploaded_files_list; source_type = "file"
            else: 
                if not image_urls_input_str.strip():
                    st.error("❌ Please enter at least one image URL."); st.stop()
                urls = [url.strip() for url in image_urls_input_str.split('\n') if url.strip()]
                if not urls:
                    st.error("❌ No valid URLs provided."); st.stop()
                image_sources_to_process = urls; source_type = "url"

            if image_sources_to_process:
                st.info(f"🧠 Processing {len(image_sources_to_process)} image source(s)...")
                progress_bar_gen = st.progress(0); status_text_gen = st.empty()
                alt_text_results = []
                current_custom_prompt = st.session_state.custom_prompt
                current_max_tokens = st.session_state.custom_max_tokens

                for i, source_item in enumerate(image_sources_to_process):
                    filename_display = ""; image_bytes = None
                    try:
                        if source_type == "file":
                            filename_display = source_item.name
                            status_text_gen.text(f"Processing: {filename_display}...")
                            image_bytes = source_item.getvalue()
                        else: 
                            filename_display = source_item 
                            status_text_gen.text(f"Fetching & Processing: {source_item[:70]}...")
                            try:
                                response = requests.get(source_item, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                                response.raise_for_status()
                                content_type = response.headers.get('content-type', '').lower()
                                if not content_type.startswith('image/'):
                                    raise ValueError(f"Content-type '{content_type}' is not an image.")
                                image_bytes = response.content
                            except requests.exceptions.RequestException as req_e:
                                st.warning(f"Skipping '{filename_display}': Fetch error - {req_e}")
                                alt_text_results.append({"Filename": filename_display, "Preview": None, "ImageIndex": i, "English Alt Text": f"Error: Fetch failed. {req_e}"})
                                progress_bar_gen.progress((i + 1) / len(image_sources_to_process)); continue
                            except ValueError as val_e:
                                st.warning(f"Skipping '{filename_display}': Invalid image - {val_e}")
                                alt_text_results.append({"Filename": filename_display, "Preview": None, "ImageIndex": i, "English Alt Text": f"Error: Invalid image. {val_e}"})
                                progress_bar_gen.progress((i + 1) / len(image_sources_to_process)); continue
                        if image_bytes:
                            thumbnail_b64 = get_thumbnail_base64(image_bytes)
                            english_alt_text = generate_alt_text_google(image_bytes, current_custom_prompt, current_max_tokens)
                            alt_text_results.append({"Filename": filename_display, "Preview": thumbnail_b64, "ImageIndex": i, "English Alt Text": english_alt_text})
                        else: 
                             alt_text_results.append({"Filename": filename_display, "Preview": None, "ImageIndex": i, "English Alt Text": "Error: No image data."})
                    except Exception as e_outer: 
                        st.error(f"Error with {filename_display}: {e_outer}")
                        alt_text_results.append({"Filename": filename_display, "Preview": None, "ImageIndex": i, "English Alt Text": f"Error: {e_outer}"})
                    progress_bar_gen.progress((i + 1) / len(image_sources_to_process))
                status_text_gen.success("✅ Alt text generation complete!")
                st.session_state.generated_results = pd.DataFrame(alt_text_results)
                st.session_state.translated_results = None 
                st.rerun()

    # --- Display logic for generated and translated results ---
    if st.session_state.translated_results is not None and not st.session_state.translated_results.empty:
        # If translations exist, use tabs
        st.subheader("📜 Generated Alt Texts & Translations")
        tab_english, tab_translations = st.tabs(["📝 English Source Texts", "🌐 All Translations"])

        with tab_english:
            st.markdown("View and edit the original English alt texts below.")
            df_eng_results = st.session_state.generated_results # Should be the source DF for edits
            if df_eng_results is not None and not df_eng_results.empty:
                for i_df_idx in range(len(df_eng_results)):
                    row = df_eng_results.iloc[i_df_idx]
                    with st.container(border=True): # Card-like container
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if row["Preview"]:
                                st.image(row["Preview"], width=150, caption=f"Index: {row['ImageIndex']}")
                            else:
                                st.markdown(f"*No preview for {row['Filename']}*")
                        with col2:
                            st.write(f"**{row['Filename']}**")
                            text_area_key = f"alt_text_edit_{row['ImageIndex']}"
                            st.text_area(f"Alt Text (Index: {row['ImageIndex']}):", value=row["English Alt Text"], height=100,
                                         key=text_area_key, on_change=update_alt_text_in_df,
                                         args=('generated_results', row['ImageIndex'], text_area_key),
                                         help="Edit the alt text here. Changes are saved to the current session.")
                        st.markdown("---") # Visual separator inside container
                
                col1_actions_eng, col2_actions_eng = st.columns(2)
                with col1_actions_eng:
                    st.download_button(label="📥 Download English Texts (Excel)", data=to_excel(df_eng_results),
                                       file_name="english_alt_texts.xlsx", mime="application/vnd.ms-excel", key="dl_eng_alt_tab")
                with col2_actions_eng:
                    if st.button("🌍 Translate Again / New Languages", type="secondary", key="translate_btn_tab"):
                        st.session_state.show_translation_options = True
                        # Potentially rerun or manage state to show options correctly within the tab
                        if 'show_translation_options' not in st.session_state or not st.session_state.show_translation_options:
                             st.rerun()


            if st.session_state.get('show_translation_options', False): # Translation UI
                st.markdown("---")
                st.subheader("🌐 Translation Options")
                selected_languages_names_gen = st.multiselect(
                    "Choose target languages for translation:",
                    options=list(DEEPL_EUROPEAN_LANGUAGES.keys()),
                    key="lang_multiselect_gen_mode_tab"
                )
                if st.button("🔄 Start Translation", type="primary", key="start_translation_btn_tab"):
                    if not st.session_state.get('deepl_api_key'):
                        st.error("❌ DeepL API Key needed. Enter it in the sidebar.")
                    elif not selected_languages_names_gen:
                        st.error("❌ Please select at least one language.")
                    else:
                        st.info(f"🌍 Translating into {len(selected_languages_names_gen)} language(s)...")
                        df_to_trans = st.session_state.generated_results.copy() # Use the current state of generated_results
                        prog_bar_trans = st.progress(0); stat_text_trans = st.empty()
                        total_ops = len(df_to_trans) * len(selected_languages_names_gen); ops_done = 0
                        for lang_name in selected_languages_names_gen:
                            lang_code = DEEPL_EUROPEAN_LANGUAGES[lang_name]
                            col_name = f"Alt Text - {lang_name} ({lang_code.upper()})"
                            current_translations = []
                            for _, row_trans in df_to_trans.iterrows():
                                stat_text_trans.text(f"Translating '{row_trans['Filename']}' to {lang_name}...")
                                translated = translate_text_deepl(st.session_state.deepl_api_key, row_trans["English Alt Text"], lang_code)
                                current_translations.append(translated); ops_done += 1
                                if total_ops > 0: prog_bar_trans.progress(ops_done / total_ops)
                            df_to_trans[col_name] = current_translations
                        stat_text_trans.success("✅ Translations complete!")
                        st.session_state.translated_results = df_to_trans
                        st.session_state.show_translation_options = False; st.rerun()
        
        with tab_translations:
            st.markdown("View all generated translations below.")
            df_trans_results = st.session_state.translated_results
            for i_trans_disp in range(len(df_trans_results)):
                row_trans_disp = df_trans_results.iloc[i_trans_disp]
                with st.container(border=True): # Card-like container
                    col1_td, col2_td = st.columns([1, 3])
                    with col1_td:
                        if row_trans_disp["Preview"]:
                            st.image(row_trans_disp["Preview"], width=150, caption=f"Index: {row_trans_disp['ImageIndex']}")
                    with col2_td:
                        st.write(f"**{row_trans_disp['Filename']}**")
                        st.text_area(f"English Alt Text:", value=row_trans_disp["English Alt Text"], height=80,
                                   key=f"eng_alt_trans_disp_{row_trans_disp['ImageIndex']}", disabled=True)
                        for col_disp in df_trans_results.columns:
                            if col_disp.startswith("Alt Text -") and col_disp != "English Alt Text":
                                st.text_area(f"{col_disp.replace('Alt Text - ', '')}:", value=row_trans_disp[col_disp], height=80,
                                           key=f"trans_disp_{col_disp}_{row_trans_disp['ImageIndex']}", disabled=True)
                    st.markdown("---") # Visual separator inside container
            st.download_button(label="📥 Download All Translations (Excel)", data=to_excel(df_trans_results),
                               file_name="alt_texts_with_translations.xlsx", mime="application/vnd.ms-excel", key="dl_all_trans_tab")

    elif st.session_state.generated_results is not None and not st.session_state.generated_results.empty:
        # Only English results, no translations yet
        st.subheader("📜 Generated English Alt Texts")
        st.caption("You can edit the alt text below. Changes are saved automatically to this session.")
        df_results = st.session_state.generated_results
        for i_df_idx in range(len(df_results)):
            row = df_results.iloc[i_df_idx]
            with st.container(border=True): # Card-like container
                col1, col2 = st.columns([1, 3])
                with col1:
                    if row["Preview"]:
                        st.image(row["Preview"], width=150, caption=f"Index: {row['ImageIndex']}")
                    else:
                        st.markdown(f"*No preview for {row['Filename']}*")
                with col2:
                    st.write(f"**{row['Filename']}**")
                    text_area_key = f"alt_text_edit_{row['ImageIndex']}"
                    st.text_area(f"Alt Text (Index: {row['ImageIndex']}):", value=row["English Alt Text"], height=100,
                                 key=text_area_key, on_change=update_alt_text_in_df,
                                 args=('generated_results', row['ImageIndex'], text_area_key),
                                 help="Edit the alt text here. Changes are saved to the current session.")
                st.markdown("---") # Visual separator inside container

        col1_actions, col2_actions = st.columns(2)
        with col1_actions:
            st.download_button(label="📥 Download Edited English Texts (Excel)", data=to_excel(st.session_state.generated_results),
                               file_name="generated_english_alt_texts.xlsx", mime="application/vnd.ms-excel", key="dl_eng_alt_main")
        with col2_actions:
            if st.button("🌍 Translate to Other Languages", type="secondary", key="translate_btn_main"):
                st.session_state.show_translation_options = True; st.rerun()
        
        if st.session_state.get('show_translation_options', False): # Translation UI
            st.markdown("---")
            st.subheader("🌐 Translation Options")
            selected_languages_names_gen = st.multiselect(
                "Choose target languages for translation:",
                options=list(DEEPL_EUROPEAN_LANGUAGES.keys()),
                key="lang_multiselect_gen_mode_main"
            )
            if st.button("🔄 Start Translation", type="primary", key="start_translation_btn_main"):
                if not st.session_state.get('deepl_api_key'):
                    st.error("❌ DeepL API Key needed. Enter it in the sidebar.")
                elif not selected_languages_names_gen:
                    st.error("❌ Please select at least one language.")
                else:
                    st.info(f"🌍 Translating into {len(selected_languages_names_gen)} language(s)...")
                    # Make sure to use the potentially edited generated_results
                    df_to_trans = st.session_state.generated_results.copy()
                    prog_bar_trans = st.progress(0); stat_text_trans = st.empty()
                    total_ops = len(df_to_trans) * len(selected_languages_names_gen); ops_done = 0
                    for lang_name in selected_languages_names_gen:
                        lang_code = DEEPL_EUROPEAN_LANGUAGES[lang_name]
                        col_name = f"Alt Text - {lang_name} ({lang_code.upper()})" # Ensure this doesn't clash if 'English Alt Text' is target
                        current_translations = []
                        for _, row_trans in df_to_trans.iterrows(): # Use df_to_trans
                            stat_text_trans.text(f"Translating '{row_trans['Filename']}' to {lang_name}...")
                            translated = translate_text_deepl(st.session_state.deepl_api_key, row_trans["English Alt Text"], lang_code)
                            current_translations.append(translated); ops_done += 1
                            if total_ops > 0: prog_bar_trans.progress(ops_done / total_ops)
                        # Add new translated column to df_to_trans
                        df_to_trans[col_name] = current_translations
                    stat_text_trans.success("✅ Translations complete!")
                    st.session_state.translated_results = df_to_trans # This DataFrame now includes original + translations
                    st.session_state.show_translation_options = False; st.rerun()

elif app_mode == "Translate Existing Alt Text from Excel File":
    st.header("🌍 Translate Existing Alt Text from Excel File")
    st.markdown("Upload an Excel file, specify the column containing text to translate, and choose your target languages.")

    uploaded_excel_file = st.file_uploader("Upload Excel (.xlsx):", type=["xlsx"], key="excel_uploader_trans_mode")
    
    col_excel1, col_excel2 = st.columns(2)
    with col_excel1:
        source_column_name = st.text_input("Name of column with text to translate:", value="English Alt Text", key="source_col_trans_mode", help="Case-sensitive column name from your Excel file.")
    with col_excel2:
        selected_languages_names_excel = st.multiselect(
        "Choose target languages:", options=list(DEEPL_EUROPEAN_LANGUAGES.keys()), key="lang_multi_trans_mode"
    )

    if st.button("🌐 Translate Excel File", type="primary", key="translate_excel_button_main", use_container_width=True):
        if not st.session_state.get('deepl_api_key'):
            st.error("❌ DeepL API Key required. Please enter it in the sidebar.")
        elif not uploaded_excel_file: st.error("❌ Please upload an Excel file.")
        elif not source_column_name.strip(): st.error("❌ Specify the source column name.")
        elif not selected_languages_names_excel: st.error("❌ Select at least one language.")
        else:
            st.info(f"🔄 Processing Excel file for translation...")
            try:
                excel_bytes = uploaded_excel_file.getvalue() 
                df_excel = pd.read_excel(io.BytesIO(excel_bytes)) 
                if source_column_name not in df_excel.columns:
                    st.error(f"❌ Column '{source_column_name}' not found. Available: {', '.join(df_excel.columns)}")
                else:
                    df_excel_trans = df_excel.copy()
                    prog_bar_excel = st.progress(0); stat_text_excel = st.empty()
                    total_excel_ops = len(df_excel_trans) * len(selected_languages_names_excel); ops_done_excel = 0
                    
                    for lang_name in selected_languages_names_excel:
                        lang_code = DEEPL_EUROPEAN_LANGUAGES[lang_name]
                        new_col_name = f"Translated - {lang_name} ({lang_code.upper()})"
                        current_translations = []
                        stat_text_excel.text(f"Translating column '{source_column_name}' to {lang_name}...")
                        for idx, row in df_excel_trans.iterrows():
                            text_to_translate = str(row[source_column_name]) if pd.notna(row[source_column_name]) else ""
                            translated = translate_text_deepl(st.session_state.deepl_api_key, text_to_translate, lang_code)
                            current_translations.append(translated); ops_done_excel +=1
                            if total_excel_ops > 0: prog_bar_excel.progress(ops_done_excel / total_excel_ops)
                        df_excel_trans[new_col_name] = current_translations
                    
                    stat_text_excel.success("✅ Excel translation complete!")
                    st.session_state.excel_translated_df_display = df_excel_trans 
            except Exception as e: st.error(f"💥 Error processing Excel: {type(e).__name__} - {e}")

    if 'excel_translated_df_display' in st.session_state and st.session_state.excel_translated_df_display is not None:
        st.subheader("📊 Translated Excel Data Preview")
        df_to_display = st.session_state.excel_translated_df_display
        
        # Make all columns appear as text, ensure they are not directly editable in data_editor
        column_config_display = {col: st.column_config.TextColumn(col) for col in df_to_display.columns}
        st.data_editor(df_to_display, column_config=column_config_display, hide_index=True, use_container_width=True, disabled=list(df_to_display.columns))
        
        download_filename = f"translated_{uploaded_excel_file.name}" if uploaded_excel_file and hasattr(uploaded_excel_file, 'name') else "translated_excel_output.xlsx"
        st.download_button(label="📥 Download Translated Excel File", data=to_excel(df_to_display),
            file_name=download_filename, mime="application/vnd.ms-excel", key="dl_trans_excel", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("🖼️ AI Alt Text Generator & Translator")
st.sidebar.caption("Version 1.0 | [bilal-karagoz/alt_text](https://github.com/bilal-karagoz/alt_text/)") # Added link format
