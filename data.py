import anthropic
import pandas as pd
import zipfile
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import streamlit as st
import os
load_dotenv()

# Streamlit Cloud: read from st.secrets if env var not set
if not os.getenv("ANTHROPIC_API_KEY") and hasattr(st, "secrets"):
    os.environ["ANTHROPIC_API_KEY"] = st.secrets.get("ANTHROPIC_API_KEY", "")
# Load the CSV into a dataframe
os.makedirs("extracted_data", exist_ok=True)
csv_file = "extracted_data/expense_data_1.csv"
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
    data = data[["Date", "Category", "Note", "Amount", "Income/Expense"]]
else:
    data = pd.DataFrame(columns=["Date", "Note", "Amount", "Category"])


def add_expense(data, date, category, note, amount, exp_type):
    new_entry = {
        'Date': date,
        'Category': category,
        'Note': note,
        'Amount': amount,
        'Income/Expense': exp_type
    }
    data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
    # print(f'New entry added: {note} - {amount} on {date}')
    return data


def recent_expenses(n=5):
    return data.tail(n)

def summarize_expenses(by='Category'):
    summary = data[data['Income/Expense'] == 'Expense'].groupby(by)['Amount'].sum()
    return summary.sort_values(ascending=False)

# print(summarize_expenses().dtype)
client = anthropic.Anthropic()

VALID_CATEGORIES = ["Food", "Transportation", "Entertainment", "Other"]

SYSTEM_PROMPT = """You are an expense categorizer. Given a note describing an expense, categorize it into exactly one of these six categories:
- Food: restaurants, coffee, snacks, groceries, meals, drinks, food items
- Transportation: metro, taxi, bus, fuel, parking, rideshare, travel costs
- Entertainment: movies, games, sports, concerts, outings, social events, activities
- Utilities : gas bill, water bill, electricity bill
- Shopping : new clothes, shoes and everything related to clothing
- Other: anything that does not clearly fit the above categories


Respond with only the category name. Nothing else."""


def categorize_expense(note: str) -> str:
    try:
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Note: {note}"
                }
            ]
        )
        category = message.content[0].text.strip().title()
        return category if category in VALID_CATEGORIES else "Other"
    except anthropic.APIStatusError as e:
        print(f"  API error ({e.status_code}) for '{note}': {e.message}")
        return "Other"
    except anthropic.APIConnectionError:
        print(f"  Network error for '{note}' — check your internet connection.")
        return "Other"
    except Exception as e:
        print(f"  Unexpected error for '{note}': {e}")
        return "Other"


data['Category'] = data.apply(
    lambda row: categorize_expense(row['Note'])
    if pd.isna(row['Category']) and pd.notna(row['Note'])
    else row['Category'],
    axis=1
)


st.title("Smart Expense Tracker")

with st.form("expense_form"):
    date = st.date_input("Date")
    note = st.text_input("Description *")
    amount = st.number_input("Amount *", min_value=0.0, format="%.2f")

    predicted_category = ""
    if note:
        predicted_category = categorize_expense(note)

    category = st.text_input(
        "Category (auto-predicted, but you can edit)", 
        value=predicted_category
    )

    submitted = st.form_submit_button("Add Expense")

    if submitted:
        if not note or not amount:
            st.error("Please fill out all required fields marked with an asterisk (*).")
        else:
            st.success(f"Form processed successfully!")
            new_expense = {"Date": date, "Note": note, "Amount": amount, "Category": category}
            data = pd.concat([data, pd.DataFrame([new_expense])], ignore_index=True)
            data.to_csv(csv_file, index=False)
            st.success(f"Added: {note} - {amount} ({category})")

st.subheader("All Expenses")
st.dataframe(data)

if not data.empty:
    st.subheader("Expense Breakdown by Category")

    category_totals = data.groupby("Category")["Amount"].sum()

    # Bar Chart
    fig, ax = plt.subplots()
    category_totals.plot(kind="bar", ax=ax)
    ax.set_ylabel("Amount")
    st.pyplot(fig)

    # Pie Chart
    st.subheader("Category Distribution")
    fig2, ax2 = plt.subplots()
    category_totals.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
    st.pyplot(fig2)
