"""
test_document_loader.py

Comprehensive validation script for the Intelligent Hospital Information
Assistant Document Loader.

This script validates:

1. Every individual parser
2. Sample page_content
3. Sample metadata
4. load_all_documents()
5. Total document consistency

Run:

    python test_document_loader.py
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.document_loader import DocumentLoader


def validate_parser(name: str, parser):
    """Validate a single parser."""

    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    try:
        documents = parser()

        print(f"Status               : SUCCESS")
        print(f"Documents Loaded     : {len(documents)}")

        if documents:
            print("\nSample Page Content")
            print("-" * 80)
            print(documents[0].page_content)

            print("\nSample Metadata")
            print("-" * 80)
            print(documents[0].metadata)

        return len(documents)

    except Exception as error:
        print(f"Status               : FAILED")
        print(f"Error                : {error}")
        return 0


def main():

    loader = DocumentLoader(Path("."))

    print("\n")
    print("=" * 80)
    print("DOCUMENT LOADER VALIDATION")
    print("=" * 80)

    total = 0

    # ------------------------------------------------------------------
    # Structured Datasets
    # ------------------------------------------------------------------

    total += validate_parser(
        "Doctor Dataset",
        loader._parse_doctor_dataset,
    )

    total += validate_parser(
        "Department Dataset",
        loader._parse_department_dataset,
    )

    total += validate_parser(
        "Symptom Dataset",
        loader._parse_symptom_dataset,
    )

    total += validate_parser(
        "Disease Dataset",
        loader._parse_disease_dataset,
    )

    total += validate_parser(
        "Medicine Dataset",
        loader._parse_medicine_dataset,
    )

    total += validate_parser(
        "Navigation Dataset",
        loader._parse_navigation_dataset,
    )

    total += validate_parser(
        "Appointment Dataset",
        loader._parse_appointment_dataset,
    )

    total += validate_parser(
        "Insurance Dataset",
        loader._parse_insurance_dataset,
    )

    total += validate_parser(
        "Emergency Dataset",
        loader._parse_emergency_dataset,
    )

    # ------------------------------------------------------------------
    # Unstructured Datasets
    # ------------------------------------------------------------------

    total += validate_parser(
        "Hospital Information Dataset",
        loader._parse_hospital_information_dataset,
    )

    total += validate_parser(
        "FAQ Dataset",
        loader._parse_faq_dataset,
    )

    total += validate_parser(
        "Patient Guideline Dataset",
        loader._parse_patient_guideline_dataset,
    )

    total += validate_parser(
        "Billing Information Dataset",
        loader._parse_billing_information_dataset,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("INDIVIDUAL PARSER SUMMARY")
    print("=" * 80)

    print(f"Total Documents From Individual Parsers : {total}")

    print("\n")
    print("=" * 80)
    print("VALIDATING load_all_documents()")
    print("=" * 80)

    try:

        all_documents = loader.load_all_documents()

        print("Status               : SUCCESS")
        print(f"Total Documents      : {len(all_documents)}")

        if len(all_documents) == total:

            print("\nFINAL RESULT")
            print("✅ VALIDATION PASSED")

        else:

            print("\nFINAL RESULT")
            print("❌ VALIDATION FAILED")
            print(
                f"Mismatch detected.\n"
                f"Individual Parsers : {total}\n"
                f"load_all_documents : {len(all_documents)}"
            )

    except Exception as error:

        print("Status               : FAILED")
        print(error)


if __name__ == "__main__":
    main()