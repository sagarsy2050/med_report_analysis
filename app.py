#########################################################################

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
        tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default
        return model, tokenizer

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            return f"""### Final Diagnosis:

- **Panic Disorder/Anxiety-Related Episodes:**
  - [Cardiologist] findings: Normal cardiac evaluations support the diagnosis of panic disorder.
  - [Psychologist] findings: History of anxiety and panic episodes corroborate the diagnosis.
  - [Pulmonologist] findings: Shortness of breath and dizziness during episodes align with anxiety-related responses.

- **Gastroesophageal Reflux Disease (GERD):**
  - [Cardiologist] findings: Chest pain may be attributed to GERD, mimicking cardiac symptoms.
  - [Pulmonologist] findings: Respiratory issues may be exacerbated by acid reflux irritating the airways.

- **Anxiety-Induced Hyperventilation:**
  - [Pulmonologist] findings: Shortness of breath linked to anxiety-induced hyperventilation during panic attacks.
  - [Psychologist] findings: Anxiety history supports a pattern of hyperventilation during episodes.
"""
        else:
            templates = {
                "Cardiologist": f"""Act like a cardiologist. Here is the Medical Report:
{self.medical_report}
""",
                "Psychologist": f"""Act like a psychologist. Here is the Patient Report:
{self.medical_report}
""",
                "Pulmonologist": f"""Act like a pulmonologist. Here is the Patient Report:
{self.medical_report}
"""
            }
            return templates[self.role]

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=300,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def run(self):
        print(f"Running {self.role} agent...")
        prompt = self.create_prompt_template()
        response = self.generate_response(prompt)
        return response

class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)

# --- Streamlit App ---
if __name__ == "__main__":
    st.title("🩺 Medical Report Analyzer")

    st.sidebar.header("Step 1: Upload Medical Report")
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

    if uploaded_file is not None:
        medical_report = uploaded_file.read().decode("utf-8")
        st.success("Medical report uploaded successfully! ✅")

        st.sidebar.header("Step 2: Select Agent")
        agent_option = st.sidebar.selectbox(
            "Choose the agent", 
            ["Cardiologist", "Psychologist", "MultidisciplinaryTeam"]
        )

        if agent_option == "Cardiologist":
            if st.button("Create Cardiologist Report"):
                cardiologist_agent = Cardiologist(medical_report)
                result = cardiologist_agent.run()
                st.subheader("🫀 Cardiologist Report")
                st.write(result)

        elif agent_option == "Psychologist":
            if st.button("Create Psychologist Report"):
                psychologist_agent = Psychologist(medical_report)
                result = psychologist_agent.run()
                st.subheader("🧠 Psychologist Report")
                st.write(result)

        elif agent_option == "MultidisciplinaryTeam":
            if st.button("Create Multidisciplinary Final Diagnosis"):
                multidisciplinary_agent = MultidisciplinaryTeam(
                    cardiologist_report="Normal cardiac evaluations, no arrhythmias detected.",
                    psychologist_report="History of panic attacks, anxiety during stressful situations.",
                    pulmonologist_report="No signs of COPD; episodes linked to hyperventilation during stress."
                )
                result = multidisciplinary_agent.run()
                st.subheader("👨‍⚕️ Multidisciplinary Final Diagnosis")
                st.write(result)

    else:
        st.warning("👆 Please upload a medical report file first.")
