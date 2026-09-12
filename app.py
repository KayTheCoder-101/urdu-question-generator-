import streamlit as st
from decode import greedy_decode, beam_search

st.set_page_config(page_title="Urdu Question Generator")

st.title("Urdu Question Generator")

st.write(
    "Urdu sentence paste karein, phir jis hissay ka sawal banana hai "
    "usay neeche select/type karein taake <ans> tags lag sakein."
)

sentence = st.text_area("Urdu sentence", height=100)
answer = st.text_input("Answer span (jo hissa <ans> mein wrap hoga)")

beam_width = st.slider("Beam width", min_value=2, max_value=5, value=3)

if st.button("Generate Question"):

    if not sentence.strip() or not answer.strip():
        st.warning("Sentence aur answer dono fill karein.")

    elif answer not in sentence:
        st.warning("Answer sentence ke andar nahi mila. Text exact match hona chahiye.")

    else:
        context = sentence.replace(answer, f"<ans> {answer} </ans>", 1)

        st.subheader("Context (with answer marked)")
        st.write(context)

        with st.spinner("Generating..."):
            greedy_question = greedy_decode(context)
            beam_question = beam_search(context, beam_width=beam_width)

        st.subheader("Greedy Search")
        st.write(greedy_question)

        st.subheader("Beam Search")
        st.write(beam_question)