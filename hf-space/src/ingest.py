import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def row_to_text_airline_losses(row):
    return (
        f"{row['airline']} ({row['country']}) faces an estimated daily financial loss of "
        f"${row['estimated_daily_loss_usd']:,.0f} USD due to the Iran-US conflict. "
        f"{int(row['cancelled_flights'])} flights were cancelled and {int(row['rerouted_flights'])} were rerouted, "
        f"incurring ${row['additional_fuel_cost_usd']:,.0f} in additional fuel costs. "
        f"Approximately {int(row['passengers_impacted']):,} passengers were impacted."
    )


def row_to_text_airport_disruptions(row):
    return (
        f"{row['airport']} ({row['iata']}/{row['icao']}) in {row['country']} "
        f"(lat: {row['latitude']}, lon: {row['longitude']}): "
        f"{int(row['flights_cancelled'])} flights cancelled, {int(row['flights_delayed'])} delayed, "
        f"{int(row['flights_diverted'])} diverted. Runway status: {row['runway_status']}."
    )


def row_to_text_airspace_closures(row):
    return (
        f"{row['country']} - {row['region']}: Airspace closed from {row['closure_start_time']} "
        f"to {row['closure_end_time']}. Reason: {row['closure_reason']}. "
        f"Authority: {row['authority']}. NOTAM: {row['NOTAM_reference']}."
    )


def row_to_text_conflict_events(row):
    return (
        f"On {row['date']} at {row['time_utc']} UTC, at {row['location']} "
        f"(lat: {row['latitude']}, lon: {row['longitude']}): {row['event_type']}. "
        f"Aviation impact: {row['aviation_impact']}. Severity: {row['severity']}. "
        f"Source: {row['source']}."
    )


def row_to_text_flight_cancellations(row):
    return (
        f"On {row['date']}, {row['airline']} flight {row['flight_number']} "
        f"({row['aircraft_type']}) from {row['origin']} to {row['destination']} "
        f"at {row['airport']} ({row['country']}) was cancelled. "
        f"Reason: {row['cancellation_reason']}."
    )


def row_to_text_flight_reroutes(row):
    return (
        f"Flight {row['flight_id']} by {row['airline']}: rerouted from "
        f"'{row['original_route']}' to '{row['diverted_route']}'. "
        f"Additional distance: {int(row['additional_distance_km'])} km, "
        f"extra fuel cost: ${row['additional_fuel_cost_usd']:,.0f}, "
        f"delay: {int(row['delay_minutes'])} minutes."
    )


FILE_CONVERTERS = {
    "airline_losses_estimate.csv": ("airline_losses", row_to_text_airline_losses),
    "airport_disruptions.csv": ("airport_disruptions", row_to_text_airport_disruptions),
    "airspace_closures.csv": ("airspace_closures", row_to_text_airspace_closures),
    "conflict_events.csv": ("conflict_events", row_to_text_conflict_events),
    "flight_cancellations.csv": ("flight_cancellations", row_to_text_flight_cancellations),
    "flight_reroutes.csv": ("flight_reroutes", row_to_text_flight_reroutes),
}


def load_documents():
    documents = []
    for filename, (category, converter) in FILE_CONVERTERS.items():
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            text = converter(row)
            metadata = {"source": filename, "category": category}
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    metadata[col] = str(val) if not isinstance(val, (int, float)) else val
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Indexed {len(documents)} documents into ChromaDB at {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    print("Loading and converting CSV data to documents...")
    docs = load_documents()
    print(f"Created {len(docs)} document chunks from 6 CSV files")
    print("Building vector store (this may take a minute on first run)...")
    build_vector_store(docs)
    print("Done! Vector store is ready.")
