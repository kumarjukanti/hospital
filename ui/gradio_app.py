import os

import gradio as gr
import requests


# Local default:
#   http://127.0.0.1:8000
#
# Docker can override this with:
#   FASTAPI_BASE_URL=http://api:8000
FASTAPI_BASE_URL = os.getenv(
    "FASTAPI_BASE_URL",
    "http://127.0.0.1:8000"
)


def call_api(endpoint: str, payload: dict):
    """
    Send a request to FastAPI and return the JSON response.
    """

    url = f"{FASTAPI_BASE_URL}{endpoint}"

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as exc:

        error = {
            "error": str(exc)
        }

        if exc.response is not None:
            error["status_code"] = exc.response.status_code
            error["response"] = exc.response.text

        return error


# ============================================================
# Risk Prediction
# ============================================================

def predict_risk_ui(
    age,
    gender,
    city,
    insurance_provider,
    chronic_flag,
    department,
    visit_type,
    doctor_id,
    length_of_stay_hours,
    days_since_registration,
    visit_frequency,
    avg_los_per_patient,
    visit_month,
    visit_dayofweek
):

    payload = {
        "age": int(age),
        "gender": gender,
        "city": city,
        "insurance_provider": insurance_provider,
        "chronic_flag": int(chronic_flag),
        "department": department,
        "visit_type": visit_type,
        "doctor_id": int(doctor_id),
        "length_of_stay_hours": float(length_of_stay_hours),
        "days_since_registration": int(days_since_registration),
        "visit_frequency": int(visit_frequency),
        "avg_los_per_patient": float(avg_los_per_patient),
        "visit_month": int(visit_month),
        "visit_dayofweek": int(visit_dayofweek)
    }

    print("Risk payload:", payload)

    result = call_api(
        "/predict/risk",
        payload
    )

    print("Risk API response:", result)

    return result


# ============================================================
# Claim Prediction
# ============================================================

def predict_claim_ui(
    age,
    gender,
    city,
    insurance_provider,
    chronic_flag,
    department,
    visit_type,
    doctor_id,
    length_of_stay_hours,
    risk_score,
    billed_amount,
    days_since_registration,
    visit_frequency,
    avg_los_per_patient,
    provider_rejection_rate,
    visit_month,
    visit_dayofweek,
    high_cost_visit_flag
):

    payload = {
        "age": int(age),
        "gender": gender,
        "city": city,
        "insurance_provider": insurance_provider,
        "chronic_flag": int(chronic_flag),
        "department": department,
        "visit_type": visit_type,
        "doctor_id": int(doctor_id),
        "length_of_stay_hours": float(length_of_stay_hours),
        "risk_score": risk_score,
        "billed_amount": float(billed_amount),
        "days_since_registration": int(days_since_registration),
        "visit_frequency": int(visit_frequency),
        "avg_los_per_patient": float(avg_los_per_patient),
        "provider_rejection_rate": float(provider_rejection_rate),
        "visit_month": int(visit_month),
        "visit_dayofweek": int(visit_dayofweek),
        "high_cost_visit_flag": int(high_cost_visit_flag)
    }

    print("Claim payload:", payload)

    result = call_api(
        "/predict/claim",
        payload
    )

    print("Claim API response:", result)

    return result


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(
    title="Healthcare ML Prediction UI"
) as demo:

    gr.Markdown(
        "# Healthcare ML Prediction UI"
    )

    gr.Markdown(
        "Use this interface to test the Risk and Claim prediction "
        "models through the FastAPI service."
    )

    # ========================================================
    # Risk Tab
    # ========================================================

    with gr.Tab("Risk Prediction"):

        with gr.Row():

            with gr.Column():

                risk_age = gr.Number(
                    label="Age",
                    value=52
                )

                risk_gender = gr.Dropdown(
                    choices=["M", "F"],
                    label="Gender",
                    value="M"
                )

                risk_city = gr.Textbox(
                    label="City",
                    value="Bangalore"
                )

                risk_insurance_provider = gr.Textbox(
                    label="Insurance Provider",
                    value="CareOne"
                )

                risk_chronic_flag = gr.Dropdown(
                    choices=[0, 1],
                    label="Chronic Flag",
                    value=1
                )

                risk_department = gr.Textbox(
                    label="Department",
                    value="Cardiology"
                )

                risk_visit_type = gr.Textbox(
                    label="Visit Type",
                    value="ER"
                )

                risk_doctor_id = gr.Number(
                    label="Doctor ID",
                    value=101
                )

            with gr.Column():

                risk_length_of_stay = gr.Number(
                    label="Length of Stay Hours",
                    value=48
                )

                risk_days_since_registration = gr.Number(
                    label="Days Since Registration",
                    value=300
                )

                risk_visit_frequency = gr.Number(
                    label="Visit Frequency",
                    value=4
                )

                risk_avg_los = gr.Number(
                    label="Average LOS per Patient",
                    value=36.5
                )

                risk_visit_month = gr.Number(
                    label="Visit Month",
                    value=3
                )

                risk_visit_dayofweek = gr.Number(
                    label="Visit Day of Week",
                    value=2
                )

        risk_button = gr.Button(
            "Predict Risk"
        )

        risk_result = gr.JSON(
            label="Risk Prediction Result"
        )

        risk_button.click(
            fn=predict_risk_ui,
            inputs=[
                risk_age,
                risk_gender,
                risk_city,
                risk_insurance_provider,
                risk_chronic_flag,
                risk_department,
                risk_visit_type,
                risk_doctor_id,
                risk_length_of_stay,
                risk_days_since_registration,
                risk_visit_frequency,
                risk_avg_los,
                risk_visit_month,
                risk_visit_dayofweek
            ],
            outputs=risk_result
        )

    # ========================================================
    # Claim Tab
    # ========================================================

    with gr.Tab("Claim Prediction"):

        with gr.Row():

            with gr.Column():

                claim_age = gr.Number(
                    label="Age",
                    value=52
                )

                claim_gender = gr.Dropdown(
                    choices=["M", "F"],
                    label="Gender",
                    value="M"
                )

                claim_city = gr.Textbox(
                    label="City",
                    value="Bangalore"
                )

                claim_insurance_provider = gr.Textbox(
                    label="Insurance Provider",
                    value="CareOne"
                )

                claim_chronic_flag = gr.Dropdown(
                    choices=[0, 1],
                    label="Chronic Flag",
                    value=1
                )

                claim_department = gr.Textbox(
                    label="Department",
                    value="Cardiology"
                )

                claim_visit_type = gr.Textbox(
                    label="Visit Type",
                    value="ER"
                )

                claim_doctor_id = gr.Number(
                    label="Doctor ID",
                    value=101
                )

                claim_length_of_stay = gr.Number(
                    label="Length of Stay Hours",
                    value=48
                )

            with gr.Column():

                claim_risk_score = gr.Dropdown(
                    choices=[
                        "Low",
                        "Medium",
                        "High"
                    ],
                    label="Risk Score",
                    value="High"
                )

                claim_billed_amount = gr.Number(
                    label="Billed Amount",
                    value=65000
                )

                claim_days_since_registration = gr.Number(
                    label="Days Since Registration",
                    value=300
                )

                claim_visit_frequency = gr.Number(
                    label="Visit Frequency",
                    value=4
                )

                claim_avg_los = gr.Number(
                    label="Average LOS per Patient",
                    value=36.5
                )

                claim_provider_rejection_rate = gr.Number(
                    label="Provider Rejection Rate",
                    value=0.257
                )

                claim_visit_month = gr.Number(
                    label="Visit Month",
                    value=3
                )

                claim_visit_dayofweek = gr.Number(
                    label="Visit Day of Week",
                    value=2
                )

                claim_high_cost_visit_flag = gr.Dropdown(
                    choices=[0, 1],
                    label="High Cost Visit Flag",
                    value=1
                )

        claim_button = gr.Button(
            "Predict Claim"
        )

        claim_result = gr.JSON(
            label="Claim Prediction Result"
        )

        claim_button.click(
            fn=predict_claim_ui,
            inputs=[
                claim_age,
                claim_gender,
                claim_city,
                claim_insurance_provider,
                claim_chronic_flag,
                claim_department,
                claim_visit_type,
                claim_doctor_id,
                claim_length_of_stay,
                claim_risk_score,
                claim_billed_amount,
                claim_days_since_registration,
                claim_visit_frequency,
                claim_avg_los,
                claim_provider_rejection_rate,
                claim_visit_month,
                claim_visit_dayofweek,
                claim_high_cost_visit_flag
            ],
            outputs=claim_result
        )


# ============================================================
# Start Gradio
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )