"""
document_loader.py

Foundation module for loading the hospital knowledge base into LangChain
Document objects for the Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities:
- Initialize knowledge base paths
- Validate project directory structure
- Provide the foundation for loading structured and unstructured datasets

This module DOES NOT:
- Parse dataset contents
- Chunk documents
- Generate embeddings
- Build the vector database
- Perform retrieval
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from json import JSONDecodeError

from langchain_core.documents import Document

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

STRUCTURED_DIR = Path("knowledge_base") / "structured"
UNSTRUCTURED_DIR = Path("knowledge_base") / "unstructured"


class DocumentLoader:
    """
    Loads hospital knowledge base documents for the RAG pipeline.

    This class is responsible for locating and validating the knowledge
    base directories. Dataset parsing will be implemented in subsequent
    development phases.

    Attributes
    ----------
    project_root : Path
        Root directory of the project.

    structured_dir : Path
        Directory containing structured JSON datasets.

    unstructured_dir : Path
        Directory containing unstructured text datasets.
    """

    def __init__(self, project_root: str | Path) -> None:
        """
        Initialize the document loader.

        Parameters
        ----------
        project_root : str | Path
            Path to the project root directory.

        Raises
        ------
        ValueError
            If project_root is empty.

        FileNotFoundError
            If required knowledge base directories are missing.
        """

        if not str(project_root).strip():
            raise ValueError("Project root path cannot be empty.")

        self.project_root = Path(project_root).expanduser()

        self.structured_dir = self.project_root / STRUCTURED_DIR
        self.unstructured_dir = self.project_root / UNSTRUCTURED_DIR

        self._validate_directories()

        logger.info("DocumentLoader initialized successfully.")
        logger.debug("Project Root      : %s", self.project_root)
        logger.debug("Structured Data   : %s", self.structured_dir)
        logger.debug("Unstructured Data : %s", self.unstructured_dir)

    # -----------------------------------------------------------------
    # Internal Helper Methods
    # -----------------------------------------------------------------

    def _validate_directories(self) -> None:
        """
        Validate that the required knowledge base directories exist.

        Raises
        ------
        FileNotFoundError
            If any required directory is missing.
        """

        required_directories = {
            "Structured Knowledge Base": self.structured_dir,
            "Unstructured Knowledge Base": self.unstructured_dir,
        }

        for name, directory in required_directories.items():
            if not directory.exists():
                raise FileNotFoundError(
                    f"{name} directory not found:\n{directory}"
                )

            if not directory.is_dir():
                raise NotADirectoryError(
                    f"{name} is not a directory:\n{directory}"
                )

        logger.info("Knowledge base directory validation completed successfully.")

    # -----------------------------------------------------------------
    # Shared Formatting Helpers
    #
    # These methods centralize the value/boolean/location formatting
    # logic that was previously duplicated inside every _format_*_document
    # method. Every formatter below reuses these instead of defining its
    # own local copies.
    # -----------------------------------------------------------------

    def _format_value(self, value: object) -> str:
        """Format an arbitrary JSON value into readable text.

        Handles ``None``, strings, booleans, numbers, lists, and nested
        dictionaries uniformly so every formatter produces consistent
        "Not specified" fallbacks and consistent list/dict rendering.
        """
        if value is None:
            return "Not specified"

        if isinstance(value, str):
            cleaned_value = value.strip()
            return cleaned_value if cleaned_value else "Not specified"

        if isinstance(value, bool):
            return "Yes" if value else "No"

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, list):
            items = [self._format_value(item) for item in value]
            items = [item for item in items if item != "Not specified"]
            return ", ".join(items) if items else "Not specified"

        if isinstance(value, dict):
            parts: list[str] = []
            for key, item in value.items():
                formatted_item = self._format_value(item)
                if formatted_item == "Not specified":
                    continue
                parts.append(f"{key.replace('_', ' ').title()}: {formatted_item}")
            return "; ".join(parts) if parts else "Not specified"

        cleaned_value = str(value).strip()
        return cleaned_value if cleaned_value else "Not specified"

    def _format_bool(self, value: object) -> str:
        """Format a boolean flag as 'Yes'/'No', falling back to `_format_value`."""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return self._format_value(value)

    def _format_location(self, location: object) -> str:
        """Format a nested ``{block, floor, room}`` location dictionary."""
        if not isinstance(location, dict):
            return self._format_value(location)

        parts: list[str] = []
        block = self._format_value(location.get("block"))
        floor = self._format_value(location.get("floor"))
        room = self._format_value(location.get("room"))

        if block != "Not specified":
            parts.append(f"Block {block}")
        if floor != "Not specified":
            parts.append(f"Floor {floor}")
        if room != "Not specified":
            parts.append(f"Room {room}")

        return ", ".join(parts) if parts else "Not specified"

    def _format_flat_location(self, block: object, floor: object, room: object) -> str:
        """Format location information supplied as separate flat fields.

        Used by datasets (e.g. navigation) that store block/floor/room as
        independent top-level keys rather than a nested dictionary.
        """
        parts: list[str] = []
        block_val = self._format_value(block)
        floor_val = self._format_value(floor)
        room_val = self._format_value(room)

        if block_val != "Not specified":
            parts.append(f"Block {block_val}")
        if floor_val != "Not specified":
            parts.append(f"Floor {floor_val}")
        if room_val != "Not specified":
            parts.append(f"Room {room_val}")

        return ", ".join(parts) if parts else "Not specified"

    def _format_availability(self, availability: object) -> str:
        """Format a weekly availability schedule into readable text.

        Expects a dictionary keyed by weekday name (e.g. ``{"Monday": "09:00-13:00"}``).
        """
        if not isinstance(availability, dict):
            return self._format_value(availability)

        week_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        lines: list[str] = []

        for day in week_order:
            schedule = availability.get(day)

            if schedule is None:
                continue

            formatted_schedule = self._format_value(schedule)

            if formatted_schedule == "Not specified":
                continue

            lines.append(f"{day}: {formatted_schedule}")

        return "\n".join(lines) if lines else "Not specified"

    def _format_experience(self, value: object) -> str:
        """Format a numeric years-of-experience value (e.g. ``22`` -> ``'22 years'``)."""
        if isinstance(value, (int, float)):
            return f"{self._format_value(value)} years"
        return self._format_value(value)

    def _format_consultation_fee(self, value: object) -> str:
        """Format a numeric consultation fee with a rupee prefix (e.g. ``1200`` -> ``'\u20b91200'``)."""
        if isinstance(value, (int, float)):
            return f"\u20b9{self._format_value(value)}"
        return self._format_value(value)

    def _format_minutes(self, value: object) -> str:
        """Format a numeric duration value in minutes (e.g. ``20`` -> ``'20 minutes'``)."""
        if isinstance(value, (int, float)):
            return f"{self._format_value(value)} minutes"
        return self._format_value(value)

    # -----------------------------------------------------------------
    # Structured Dataset Formatters
    # -----------------------------------------------------------------

    def _format_doctor_document(self, record: dict) -> str:
        """Format a doctor record into a clean semantic-search profile.

        This is the reference formatter: its output is locked and must not
        change, since it has already been validated against the doctor
        directory dataset.
        """

        full_name = self._format_value(record.get("full_name"))
        gender = self._format_value(record.get("gender"))
        designation = self._format_value(record.get("designation"))
        qualification = self._format_value(record.get("qualification"))
        department_name = self._format_value(record.get("department_name"))
        specializations = self._format_value(record.get("specializations"))
        languages = self._format_value(record.get("languages"))
        rating = self._format_value(record.get("rating"))

        experience = self._format_experience(record.get("experience_years"))
        consultation_fee = self._format_consultation_fee(record.get("consultation_fee"))
        average_consultation_time = self._format_value(record.get("average_consultation_time"))
        location = self._format_location(record.get("location"))
        appointment_required = self._format_bool(record.get("appointment_required"))
        availability = self._format_availability(record.get("availability"))

        profile_lines = [
            f"Doctor: {full_name}",
            f"Gender: {gender}",
            f"Designation: {designation}",
            f"Qualification: {qualification}",
            f"Department: {department_name}",
            f"Specializations: {specializations}",
            f"Experience: {experience}",
            f"Languages: {languages}",
            f"Consultation Fee: {consultation_fee}",
            f"Average Consultation Time: {average_consultation_time}",
            f"Location: {location}",
            f"Appointment Required: {appointment_required}",
            f"Rating: {rating}",
            "",
            "Availability:",
            availability,
        ]

        return "\n".join(profile_lines)

    def _format_department_document(self, record: dict) -> str:
        """Format a department record into a clean semantic-search profile."""

        department_name = self._format_value(record.get("department_name"))
        category = self._format_value(record.get("category"))
        description = self._format_value(record.get("description"))
        services = self._format_value(record.get("services"))
        working_hours = self._format_value(record.get("working_hours"))
        location = self._format_location(record.get("location"))
        emergency_support = self._format_bool(record.get("emergency_support"))
        appointment_required = self._format_bool(record.get("appointment_required"))

        intro = f"{department_name} is a {category} at the hospital."

        profile_lines = [
            intro,
            description,
            "",
            f"Services: {services}",
            f"Working Hours: {working_hours}",
            f"Location: {location}",
            f"Emergency Support: {emergency_support}",
            f"Appointment Required: {appointment_required}",
        ]

        return "\n".join(profile_lines)

    def _format_symptom_document(self, record: dict) -> str:
        """Format a symptom record into a clean semantic-search profile."""

        symptom_name = self._format_value(record.get("symptom_name"))
        category = self._format_value(record.get("category"))
        description = self._format_value(record.get("description"))
        severity = self._format_value(record.get("severity"))
        emergency = self._format_bool(record.get("emergency"))
        recommended_department_name = self._format_value(record.get("recommended_department_name"))
        first_aid = self._format_value(record.get("first_aid"))
        red_flags = self._format_value(record.get("red_flags"))

        intro = f"{symptom_name} is a {severity.lower() if severity != 'Not specified' else severity} severity symptom in the {category} category."

        profile_lines = [
            intro,
            description,
            "",
            f"Emergency: {emergency}",
            f"Recommended Department: {recommended_department_name}",
            f"First Aid: {first_aid}",
            f"Red Flags: {red_flags}",
        ]

        return "\n".join(profile_lines)

    def _format_disease_document(self, record: dict) -> str:
        """Format a disease record into a clean semantic-search profile."""

        disease_name = self._format_value(record.get("disease_name"))
        icd_category = self._format_value(record.get("icd_category"))
        description = self._format_value(record.get("description"))
        severity = self._format_value(record.get("severity"))
        emergency = self._format_bool(record.get("emergency"))
        department_name = self._format_value(record.get("department_name"))
        common_symptoms = self._format_value(record.get("common_symptoms"))
        diagnostic_tests = self._format_value(record.get("diagnostic_tests"))
        follow_up = self._format_value(record.get("follow_up"))

        intro = (
            f"{disease_name} is a {severity.lower() if severity != 'Not specified' else severity} "
            f"severity condition classified under {icd_category}, treated in the {department_name} department."
        )

        profile_lines = [
            intro,
            description,
            "",
            f"Emergency: {emergency}",
            f"Common Symptoms: {common_symptoms}",
            f"Diagnostic Tests: {diagnostic_tests}",
            f"Follow Up: {follow_up}",
        ]

        return "\n".join(profile_lines)

    def _format_medicine_document(self, record: dict) -> str:
        """Format a medicine record into a clean semantic-search profile."""

        generic_name = self._format_value(record.get("generic_name"))
        brand_names = self._format_value(record.get("brand_names"))
        drug_class = self._format_value(record.get("drug_class"))
        dosage_form = self._format_value(record.get("dosage_form"))
        strength = self._format_value(record.get("strength"))
        used_for = self._format_value(record.get("used_for"))
        adult_dose = self._format_value(record.get("adult_dose"))
        child_dose = self._format_value(record.get("child_dose"))
        frequency = self._format_value(record.get("frequency"))
        maximum_daily_dose = self._format_value(record.get("maximum_daily_dose"))
        food_instruction = self._format_value(record.get("food_instruction"))
        best_time = self._format_value(record.get("best_time"))
        contraindications = self._format_value(record.get("contraindications"))
        side_effects = self._format_value(record.get("side_effects"))
        storage = self._format_value(record.get("storage"))
        pregnancy_safe = self._format_bool(record.get("pregnancy_safe"))
        prescription_required = self._format_bool(record.get("prescription_required"))

        drug_class_article = "an" if drug_class[:1].lower() in "aeiou" else "a"
        intro = (
            f"{generic_name} (also sold as {brand_names}) is {drug_class_article} {drug_class} available as "
            f"{dosage_form} in {strength} strengths, commonly used for {used_for}."
        )

        profile_lines = [
            intro,
            "",
            f"Adult Dose: {adult_dose}",
            f"Child Dose: {child_dose}",
            f"Frequency: {frequency}",
            f"Maximum Daily Dose: {maximum_daily_dose}",
            f"Food Instruction: {food_instruction}",
            f"Best Time: {best_time}",
            f"Contraindications: {contraindications}",
            f"Side Effects: {side_effects}",
            f"Storage: {storage}",
            f"Pregnancy Safe: {pregnancy_safe}",
            f"Prescription Required: {prescription_required}",
        ]

        return "\n".join(profile_lines)

    def _format_navigation_document(self, record: dict) -> str:
        """Format a navigation record into a clean semantic-search profile."""

        department_name = self._format_value(record.get("department_name"))
        building = self._format_value(record.get("building"))
        location = self._format_flat_location(
            record.get("block"), record.get("floor"), record.get("room_number")
        )
        nearest_landmark = self._format_value(record.get("nearest_landmark"))
        distance_from_main_entrance = self._format_value(record.get("distance_from_main_entrance"))
        estimated_walking_time = self._format_value(record.get("estimated_walking_time"))
        operating_hours = self._format_value(record.get("operating_hours"))
        nearest_help_desk = self._format_value(record.get("nearest_help_desk"))
        nearby_departments = self._format_value(record.get("nearby_departments"))
        wheelchair_accessible = self._format_bool(record.get("wheelchair_accessible"))
        elevator_available = self._format_bool(record.get("elevator_available"))
        stair_access = self._format_bool(record.get("stair_access"))

        intro = (
            f"{department_name} is located in {building} ({location}), near {nearest_landmark}. "
            f"It is approximately {distance_from_main_entrance} from the main entrance "
            f"(about {estimated_walking_time} on foot), open {operating_hours}."
        )

        profile_lines = [
            intro,
            "",
            f"Nearest Help Desk: {nearest_help_desk}",
            f"Nearby Departments: {nearby_departments}",
            f"Wheelchair Accessible: {wheelchair_accessible}",
            f"Elevator Available: {elevator_available}",
            f"Stair Access: {stair_access}",
        ]

        return "\n".join(profile_lines)

    def _format_appointment_document(self, record: dict) -> str:
        """Format an appointment record into a clean semantic-search profile."""

        appointment_type = self._format_value(record.get("appointment_type"))
        department_name = self._format_value(record.get("department_name"))
        eligibility = self._format_value(record.get("eligibility"))
        priority = self._format_value(record.get("priority"))
        booking_modes = self._format_value(record.get("booking_modes"))
        working_days = self._format_value(record.get("working_days"))
        working_hours = self._format_value(record.get("working_hours"))
        availability = self._format_value(record.get("availability"))
        average_wait_time = self._format_minutes(record.get("average_wait_time_minutes"))
        estimated_consultation_duration = self._format_minutes(
            record.get("estimated_consultation_duration_minutes")
        )
        consultation_fee = self._format_value(record.get("consultation_fee"))
        advance_booking_required = self._format_bool(record.get("advance_booking_required"))
        walk_in_allowed = self._format_bool(record.get("walk_in_allowed"))
        requires_referral = self._format_bool(record.get("requires_referral"))
        reschedule_allowed = self._format_bool(record.get("reschedule_allowed"))
        teleconsultation_available = self._format_bool(record.get("teleconsultation_available"))
        online_payment_available = self._format_bool(record.get("online_payment_available"))
        documents_required = self._format_value(record.get("documents_required"))
        cancellation_policy = self._format_value(record.get("cancellation_policy"))
        follow_up_policy = self._format_value(record.get("follow_up_policy"))
        appointment_instructions = self._format_value(record.get("appointment_instructions"))
        contact_extension = self._format_value(record.get("contact_extension"))

        intro = (
            f"{appointment_type} in {department_name} is available for {eligibility}, "
            f"with {priority} priority. Booking modes include {booking_modes}, on "
            f"{working_days} during {working_hours}. {availability}"
        )

        profile_lines = [
            intro,
            "",
            f"Average Wait Time: {average_wait_time}",
            f"Estimated Consultation Duration: {estimated_consultation_duration}",
            f"Consultation Fee: {consultation_fee}",
            f"Advance Booking Required: {advance_booking_required}",
            f"Walk In Allowed: {walk_in_allowed}",
            f"Requires Referral: {requires_referral}",
            f"Reschedule Allowed: {reschedule_allowed}",
            f"Teleconsultation Available: {teleconsultation_available}",
            f"Online Payment Available: {online_payment_available}",
            f"Documents Required: {documents_required}",
            f"Cancellation Policy: {cancellation_policy}",
            f"Follow Up Policy: {follow_up_policy}",
            f"Appointment Instructions: {appointment_instructions}",
            f"Contact Extension: {contact_extension}",
        ]

        return "\n".join(profile_lines)

    def _format_insurance_document(self, record: dict) -> str:
        """Format an insurance record into a clean semantic-search profile."""

        provider_name = self._format_value(record.get("provider_name"))
        plan_type = self._format_value(record.get("plan_type"))
        coverage_type = self._format_value(record.get("coverage_type"))
        coverage_limit = self._format_value(record.get("coverage_limit"))
        room_rent_limit = self._format_value(record.get("room_rent_limit"))
        waiting_period = self._format_value(record.get("waiting_period"))
        cashless = self._format_bool(record.get("cashless"))
        network_hospital = self._format_bool(record.get("network_hospital"))
        pre_authorization_required = self._format_bool(record.get("pre_authorization_required"))
        covered_departments = self._format_value(record.get("covered_departments"))
        required_documents = self._format_value(record.get("required_documents"))
        exclusions = self._format_value(record.get("exclusions"))
        claim_process = self._format_value(record.get("claim_process"))
        customer_support_hours = self._format_value(record.get("customer_support_hours"))
        helpline = self._format_value(record.get("helpline"))
        contact_email = self._format_value(record.get("contact_email"))
        website = self._format_value(record.get("website"))
        remarks = self._format_value(record.get("remarks"))

        intro = (
            f"{provider_name} is a {plan_type} plan offering {coverage_type} coverage "
            f"up to {coverage_limit}, with a {room_rent_limit} room rent limit and "
            f"{waiting_period} waiting period."
        )

        profile_lines = [
            intro,
            "",
            f"Cashless: {cashless}",
            f"Network Hospital: {network_hospital}",
            f"Pre-Authorization Required: {pre_authorization_required}",
            f"Covered Departments: {covered_departments}",
            f"Required Documents: {required_documents}",
            f"Exclusions: {exclusions}",
            f"Claim Process: {claim_process}",
            f"Customer Support Hours: {customer_support_hours}",
            f"Helpline: {helpline}",
            f"Contact Email: {contact_email}",
            f"Website: {website}",
            f"Remarks: {remarks}",
        ]

        return "\n".join(profile_lines)

    def _format_emergency_document(self, record: dict) -> str:
        """Format an emergency protocol record into a clean semantic-search profile."""

        emergency_name = self._format_value(record.get("emergency_name"))
        category = self._format_value(record.get("category"))
        severity_level = self._format_value(record.get("severity_level"))
        triage_priority = self._format_value(record.get("triage_priority"))
        description = self._format_value(record.get("description"))
        common_symptoms = self._format_value(record.get("common_symptoms"))
        immediate_actions = self._format_value(record.get("immediate_actions"))
        stabilization_steps = self._format_value(record.get("stabilization_steps"))
        do_not = self._format_value(record.get("do_not"))
        required_equipment = self._format_value(record.get("required_equipment"))
        possible_complications = self._format_value(record.get("possible_complications"))
        follow_up_care = self._format_value(record.get("follow_up_care"))
        recommended_department = self._format_value(record.get("recommended_department"))
        requires_ambulance = self._format_bool(record.get("requires_ambulance"))
        call_emergency_number = self._format_value(record.get("call_emergency_number"))
        remarks = self._format_value(record.get("remarks"))

        intro = (
            f"{emergency_name} is a {severity_level.lower() if severity_level != 'Not specified' else severity_level} "
            f"severity {category} emergency with {triage_priority} triage priority."
        )

        profile_lines = [
            intro,
            description,
            "",
            f"Common Symptoms: {common_symptoms}",
            f"Immediate Actions: {immediate_actions}",
            f"Stabilization Steps: {stabilization_steps}",
            f"Do Not: {do_not}",
            f"Required Equipment: {required_equipment}",
            f"Possible Complications: {possible_complications}",
            f"Follow Up Care: {follow_up_care}",
            f"Recommended Department: {recommended_department}",
            f"Requires Ambulance: {requires_ambulance}",
            f"Call Emergency Number: {call_emergency_number}",
            f"Remarks: {remarks}",
        ]

        return "\n".join(profile_lines)

    # -----------------------------------------------------------------
    # JSON Loading Utilities
    # -----------------------------------------------------------------

    def _load_json_file(self, file_path: Path) -> list[dict] | dict:
        """Load and parse a JSON file from disk.

        The file is read with ``utf-8-sig`` encoding so UTF-8 BOM markers are
        handled transparently. The method validates that the file exists before
        reading it and logs both success and failure events.

        Args:
            file_path: Path to the JSON file.

        Returns:
            The parsed JSON object, which may be a list of dictionaries or a
            dictionary depending on the file contents.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as valid JSON.
        """
        logger.info("Starting JSON file load: %s", file_path)

        if not file_path.exists():
            logger.error("JSON file not found: %s", file_path)
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8-sig") as file_handle:
                parsed_json = json.load(file_handle)
        except JSONDecodeError as exc:
            logger.exception("Failed to decode JSON file: %s", file_path)
            raise ValueError(
                f"Invalid JSON in file '{file_path.name}': {exc.msg}"
            ) from exc
        except Exception:
            logger.exception("Failed to load JSON file: %s", file_path)
            raise

        logger.info("Successfully loaded JSON file: %s", file_path)
        return parsed_json

    # -----------------------------------------------------------------
    # Text Loading Utilities
    # -----------------------------------------------------------------

    def _load_text_file(self, file_path: Path) -> str:
        """Load a UTF-8 text file and normalize its line endings.

        The method verifies that the file exists before reading it, converts
        Windows line endings to Unix line endings, preserves paragraph spacing,
        and strips only trailing whitespace from the final content.

        Args:
            file_path: Path to the text file.

        Returns:
            The normalized file content as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        logger.info("Starting text file load: %s", file_path)

        if not file_path.exists():
            logger.error("Text file not found: %s", file_path)
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            with file_path.open("r", encoding="utf-8") as file_handle:
                file_content = file_handle.read()
        except Exception:
            logger.exception("Failed to load text file: %s", file_path)
            raise

        normalized_content = file_content.replace("\r\n", "\n").rstrip()

        logger.info("Successfully loaded text file: %s", file_path)
        return normalized_content

    def _create_document(self, page_content: str, metadata: dict) -> Document:
        """Create a LangChain document from validated content and metadata.

        This helper normalizes the supplied page content by stripping leading
        and trailing whitespace, validates that the resulting content is not
        empty, and ensures metadata is available as a dictionary before
        constructing the LangChain :class:`Document` instance.

        Args:
            page_content: Raw textual content to store in the document.
            metadata: Metadata to attach to the document.

        Returns:
            A LangChain Document instance containing the normalized content and
            metadata.

        Raises:
            ValueError: If the stripped content is empty or metadata cannot be
                converted into a dictionary.
        """
        logger.debug("Creating document with metadata keys: %s", list(metadata.keys()) if isinstance(metadata, dict) else "invalid metadata")

        normalized_content = page_content.strip()
        if not normalized_content:
            logger.error("Cannot create document from empty page content.")
            raise ValueError("page_content cannot be empty after stripping whitespace.")

        if not isinstance(metadata, dict):
            try:
                metadata = dict(metadata)
            except (TypeError, ValueError) as exc:
                logger.exception("Invalid metadata provided for document creation.")
                raise ValueError("metadata must be a dictionary or convertible to a dictionary.") from exc

        document = Document(page_content=normalized_content, metadata=metadata)
        logger.debug("Document created successfully.")
        return document

    # -----------------------------------------------------------------
    # Structured Dataset Parsers
    # -----------------------------------------------------------------

    def _parse_doctor_dataset(self) -> list[Document]:
        """Parse the doctor directory dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid doctor
            record in ``knowledge_base/structured/doctor_directory.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``doctors`` field is not a list of doctor records.
        """

        source_file = "doctor_directory.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading doctor directory dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Doctor directory dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "doctor_directory.json must contain a top-level dictionary."
            )

        doctor_records = dataset.get("doctors")
        if not isinstance(doctor_records, list):
            logger.error(
                "Doctor directory dataset must contain a 'doctors' list: %s",
                file_path,
            )
            raise ValueError(
                "doctor_directory.json must contain a 'doctors' list of doctor records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(doctor_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid doctor record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_doctor_document(record)
                metadata = {
                    "source": "doctor_directory",
                    "record_type": "doctor",
                    "source_file": source_file,
                    "doctor_id": record.get("doctor_id"),
                    "department_id": record.get("department_id"),
                    "department_name": record.get("department_name"),
                    "navigation_id": record.get("navigation_id"),
                    "experience_years": record.get("experience_years"),
                    "consultation_fee": record.get("consultation_fee"),
                    "appointment_required": record.get("appointment_required"),
                    "rating": record.get("rating"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid doctor record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid doctor records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Doctor directory parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(doctor_records),
            skipped_count,
        )
        return documents

    def _parse_department_dataset(self) -> list[Document]:
        """Parse the department master dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid
            department record in
            ``knowledge_base/structured/department_master.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``departments`` field is not a list of department records.
        """

        source_file = "department_master.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading department master dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Department master dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "department_master.json must contain a top-level dictionary."
            )

        department_records = dataset.get("departments")
        if not isinstance(department_records, list):
            logger.error(
                "Department master dataset must contain a 'departments' list: %s",
                file_path,
            )
            raise ValueError(
                "department_master.json must contain a 'departments' list of department records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(department_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid department record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_department_document(record)
                metadata = {
                    "source": "department_master",
                    "record_type": "department",
                    "source_file": source_file,
                    "department_id": record.get("department_id"),
                    "department_name": record.get("department_name"),
                    "category": record.get("category"),
                    "navigation_id": record.get("navigation_id"),
                    "head_doctor_id": record.get("head_doctor_id"),
                    "emergency_support": record.get("emergency_support"),
                    "appointment_required": record.get("appointment_required"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid department record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid department records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Department master parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(department_records),
            skipped_count,
        )
        return documents

    def _parse_symptom_dataset(self) -> list[Document]:
        """Parse the symptom mapping dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid symptom
            record in ``knowledge_base/structured/symptom_mapping.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``symptoms`` field is not a list of symptom records.
        """

        source_file = "symptom_mapping.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading symptom mapping dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Symptom mapping dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "symptom_mapping.json must contain a top-level dictionary."
            )

        symptom_records = dataset.get("symptoms")
        if not isinstance(symptom_records, list):
            logger.error(
                "Symptom mapping dataset must contain a 'symptoms' list: %s",
                file_path,
            )
            raise ValueError(
                "symptom_mapping.json must contain a 'symptoms' list of symptom records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(symptom_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid symptom record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_symptom_document(record)
                metadata = {
                    "source": "symptom_mapping",
                    "record_type": "symptom",
                    "source_file": source_file,
                    "symptom_id": record.get("symptom_id"),
                    "symptom_name": record.get("symptom_name"),
                    "category": record.get("category"),
                    "severity": record.get("severity"),
                    "emergency": record.get("emergency"),
                    "recommended_department_id": record.get("recommended_department_id"),
                    "recommended_department_name": record.get("recommended_department_name"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid symptom record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid symptom records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Symptom mapping parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(symptom_records),
            skipped_count,
        )
        return documents

    def _parse_disease_dataset(self) -> list[Document]:
        """Parse the disease mapping dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid disease
            record in ``knowledge_base/structured/disease_mapping.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``diseases`` field is not a list of disease records.
        """

        source_file = "disease_mapping.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading disease mapping dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Disease mapping dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "disease_mapping.json must contain a top-level dictionary."
            )

        disease_records = dataset.get("diseases")
        if not isinstance(disease_records, list):
            logger.error(
                "Disease mapping dataset must contain a 'diseases' list: %s",
                file_path,
            )
            raise ValueError(
                "disease_mapping.json must contain a 'diseases' list of disease records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(disease_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid disease record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_disease_document(record)
                metadata = {
                    "source": "disease_mapping",
                    "record_type": "disease",
                    "source_file": source_file,
                    "disease_id": record.get("disease_id"),
                    "disease_name": record.get("disease_name"),
                    "department_id": record.get("department_id"),
                    "department_name": record.get("department_name"),
                    "severity": record.get("severity"),
                    "emergency": record.get("emergency"),
                    "navigation_id": record.get("navigation_id"),
                    "primary_doctor_id": record.get("primary_doctor_id"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid disease record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid disease records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Disease mapping parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(disease_records),
            skipped_count,
        )
        return documents

    def _parse_medicine_dataset(self) -> list[Document]:
        """Parse the medicine database dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid medicine
            record in ``knowledge_base/structured/medicine_database.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``medicines`` field is not a list of medicine records.
        """

        source_file = "medicine_database.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading medicine database dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Medicine database dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "medicine_database.json must contain a top-level dictionary."
            )

        medicine_records = dataset.get("medicines")
        if not isinstance(medicine_records, list):
            logger.error(
                "Medicine database dataset must contain a 'medicines' list: %s",
                file_path,
            )
            raise ValueError(
                "medicine_database.json must contain a 'medicines' list of medicine records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(medicine_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid medicine record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_medicine_document(record)
                metadata = {
                    "source": "medicine_database",
                    "record_type": "medicine",
                    "source_file": source_file,
                    "medicine_id": record.get("medicine_id"),
                    "generic_name": record.get("generic_name"),
                    "drug_class": record.get("drug_class"),
                    "prescription_required": record.get("prescription_required"),
                    "pregnancy_safe": record.get("pregnancy_safe"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid medicine record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid medicine records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Medicine database parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(medicine_records),
            skipped_count,
        )
        return documents

    def _parse_navigation_dataset(self) -> list[Document]:
        """Parse the navigation dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid
            navigation record in ``knowledge_base/structured/navigation.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``navigation`` field is not a list of navigation records.
        """

        source_file = "navigation.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading navigation dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Navigation dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "navigation.json must contain a top-level dictionary."
            )

        navigation_records = dataset.get("navigation")
        if not isinstance(navigation_records, list):
            logger.error(
                "Navigation dataset must contain a 'navigation' list: %s",
                file_path,
            )
            raise ValueError(
                "navigation.json must contain a 'navigation' list of navigation records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(navigation_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid navigation record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_navigation_document(record)
                metadata = {
                    "source": "navigation",
                    "record_type": "navigation",
                    "source_file": source_file,
                    "navigation_id": record.get("navigation_id"),
                    "department_id": record.get("department_id"),
                    "department_name": record.get("department_name"),
                    "block": record.get("block"),
                    "floor": record.get("floor"),
                    "wheelchair_accessible": record.get("wheelchair_accessible"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid navigation record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid navigation records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Navigation parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(navigation_records),
            skipped_count,
        )
        return documents

    def _parse_appointment_dataset(self) -> list[Document]:
        """Parse the appointments dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid
            appointment record in
            ``knowledge_base/structured/appointments.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``appointments`` field is not a list of appointment
                records.
        """

        source_file = "appointments.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading appointments dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Appointments dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "appointments.json must contain a top-level dictionary."
            )

        appointment_records = dataset.get("appointments")
        if not isinstance(appointment_records, list):
            logger.error(
                "Appointments dataset must contain an 'appointments' list: %s",
                file_path,
            )
            raise ValueError(
                "appointments.json must contain an 'appointments' list of appointment records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(appointment_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid appointment record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_appointment_document(record)
                metadata = {
                    "source": "appointments",
                    "record_type": "appointment",
                    "source_file": source_file,
                    "appointment_id": record.get("appointment_id"),
                    "department_id": record.get("department_id"),
                    "department_name": record.get("department_name"),
                    "appointment_type": record.get("appointment_type"),
                    "priority": record.get("priority"),
                    "walk_in_allowed": record.get("walk_in_allowed"),
                    "requires_referral": record.get("requires_referral"),
                    "average_wait_time_minutes": record.get("average_wait_time_minutes"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid appointment record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid appointment records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Appointments parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(appointment_records),
            skipped_count,
        )
        return documents

    def _parse_insurance_dataset(self) -> list[Document]:
        """Parse the insurance dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid
            insurance record in ``knowledge_base/structured/insurance.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``insurance`` field is not a list of insurance records.
        """

        source_file = "insurance.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading insurance dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Insurance dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "insurance.json must contain a top-level dictionary."
            )

        insurance_records = dataset.get("insurance")
        if not isinstance(insurance_records, list):
            logger.error(
                "Insurance dataset must contain an 'insurance' list: %s",
                file_path,
            )
            raise ValueError(
                "insurance.json must contain an 'insurance' list of insurance records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(insurance_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid insurance record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_insurance_document(record)
                metadata = {
                    "source": "insurance",
                    "record_type": "insurance",
                    "source_file": source_file,
                    "insurance_id": record.get("insurance_id"),
                    "provider_name": record.get("provider_name"),
                    "plan_type": record.get("plan_type"),
                    "coverage_type": record.get("coverage_type"),
                    "cashless": record.get("cashless"),
                    "network_hospital": record.get("network_hospital"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid insurance record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid insurance records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Insurance parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(insurance_records),
            skipped_count,
        )
        return documents

    def _parse_emergency_dataset(self) -> list[Document]:
        """Parse the emergency protocols dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid
            emergency protocol record in
            ``knowledge_base/structured/emergency_protocols.json``.

        Raises:
            ValueError: If the loaded JSON payload is not a dictionary or if
                the ``emergency_protocols`` field is not a list of protocol
                records.
        """

        source_file = "emergency_protocols.json"
        file_path = self.structured_dir / source_file
        logger.info("Loading emergency protocols dataset: %s", file_path)

        dataset = self._load_json_file(file_path)
        if not isinstance(dataset, dict):
            logger.error(
                "Emergency protocols dataset must be a top-level dictionary: %s",
                file_path,
            )
            raise ValueError(
                "emergency_protocols.json must contain a top-level dictionary."
            )

        emergency_records = dataset.get("emergency_protocols")
        if not isinstance(emergency_records, list):
            logger.error(
                "Emergency protocols dataset must contain an 'emergency_protocols' list: %s",
                file_path,
            )
            raise ValueError(
                "emergency_protocols.json must contain an 'emergency_protocols' list of protocol records."
            )

        documents: list[Document] = []
        skipped_count = 0

        for index, record in enumerate(emergency_records):
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping invalid emergency protocol record at index %s: expected dict, got %s",
                    index,
                    type(record).__name__,
                )
                skipped_count += 1
                continue

            try:
                page_content = self._format_emergency_document(record)
                metadata = {
                    "source": "emergency_protocols",
                    "record_type": "emergency_protocol",
                    "source_file": source_file,
                    "protocol_id": record.get("protocol_id"),
                    "emergency_name": record.get("emergency_name"),
                    "category": record.get("category"),
                    "severity_level": record.get("severity_level"),
                    "triage_priority": record.get("triage_priority"),
                    "recommended_department": record.get("recommended_department"),
                    "requires_ambulance": record.get("requires_ambulance"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid emergency protocol record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid emergency protocol records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Emergency protocols parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(emergency_records),
            skipped_count,
        )
        return documents

    # -----------------------------------------------------------------
    # Shared Unstructured Text Helpers
    #
    # These helpers centralize the block-splitting and field-extraction
    # logic shared by every unstructured (.txt) dataset parser below, the
    # same way the structured formatting helpers above are shared by the
    # structured dataset formatters.
    # -----------------------------------------------------------------

    def _split_text_blocks(self, text: str) -> list[str]:
        """Split a raw record's text into blank-line separated blocks.

        Handles single and multiple blank lines, trailing whitespace, and
        blocks that themselves span multiple lines (e.g. multi-line list
        values such as "Related Departments").

        Args:
            text: Raw text for a single record.

        Returns:
            A list of non-empty, stripped text blocks in their original
            order.
        """
        if not text or not text.strip():
            return []

        raw_blocks = re.split(r"\n\s*\n", text.strip())
        return [block.strip() for block in raw_blocks if block.strip()]

    def _extract_record_id(self, id_block: str) -> str | None:
        """Extract a record identifier from a block, tolerating ``[brackets]``.

        Args:
            id_block: The first block of a record, expected to contain only
                the record identifier (e.g. ``"GUIDELINE-001"`` or
                ``"[HOSP-001]"``).

        Returns:
            The extracted identifier, or ``None`` if the block does not look
            like a valid identifier.
        """
        if not id_block:
            return None

        match = re.match(
            r"^\[?(?P<id>[A-Za-z0-9][A-Za-z0-9\-_]*)\]?$",
            id_block.strip(),
        )
        return match.group("id") if match else None

    def _extract_labeled_field(self, block: str) -> tuple[str, str] | None:
        """Split a ``"Label:\\nvalue..."`` block into a ``(label, value)`` pair.

        Multi-line values (e.g. a list of related department names, one per
        line) are normalized into a single comma-separated string so they
        read naturally in formatted output.

        Args:
            block: A single labeled block, e.g. ``"Category:\\nEmergency"``.

        Returns:
            A ``(label, value)`` tuple where ``label`` is lowercased and
            underscored (e.g. ``"related_departments"``), or ``None`` if the
            block does not start with a recognizable ``"Label:"`` line.
        """
        lines = block.split("\n")
        label_line = lines[0].strip()

        if not label_line.endswith(":"):
            return None

        label = label_line[:-1].strip().lower().replace(" ", "_")
        value_lines = [line.strip() for line in lines[1:] if line.strip()]
        value = ", ".join(value_lines)

        return label, value

    # -----------------------------------------------------------------
    # Unstructured Dataset Formatters
    # -----------------------------------------------------------------

    def _format_hospital_information_document(self, record: dict) -> str:
        """Format a hospital information record into a clean semantic-search profile."""

        record_id = self._format_value(record.get("record_id"))
        category = self._format_value(record.get("category"))
        title = self._format_value(record.get("title"))
        description = self._format_value(record.get("description"))
        location = self._format_value(record.get("location"))
        operating_hours = self._format_value(record.get("operating_hours"))
        contact = self._format_value(record.get("contact"))
        related_departments = self._format_value(record.get("related_departments"))
        keywords = self._format_value(record.get("keywords"))
        related_records = self._format_value(record.get("related_records"))

        intro = f"{title} ({record_id}) is a hospital information entry in the {category} category."

        profile_lines = [
            intro,
            description,
            "",
            f"Location: {location}",
            f"Operating Hours: {operating_hours}",
            f"Contact: {contact}",
            f"Related Departments: {related_departments}",
            f"Keywords: {keywords}",
            f"Related Records: {related_records}",
        ]

        return "\n".join(profile_lines)

    def _format_faq_document(self, record: dict) -> str:
        """Format a FAQ record into a clean semantic-search profile."""

        record_id = self._format_value(record.get("record_id"))
        question = self._format_value(record.get("question"))
        answer = self._format_value(record.get("answer"))

        profile_lines = [
            f"Frequently Asked Question ({record_id}): {question}",
            "",
            answer,
        ]

        return "\n".join(profile_lines)

    def _format_patient_guideline_document(self, record: dict) -> str:
        """Format a patient guideline record into a clean semantic-search profile."""

        record_id = self._format_value(record.get("record_id"))
        title = self._format_value(record.get("title"))
        guideline = self._format_value(record.get("guideline"))

        profile_lines = [
            f"Patient Guideline ({record_id}): {title}",
            "",
            guideline,
        ]

        return "\n".join(profile_lines)

    def _format_billing_information_document(self, record: dict) -> str:
        """Format a billing information record into a clean semantic-search profile."""

        record_id = self._format_value(record.get("record_id"))
        title = self._format_value(record.get("title"))
        information = self._format_value(record.get("information"))

        profile_lines = [
            f"Billing Information ({record_id}): {title}",
            "",
            information,
        ]

        return "\n".join(profile_lines)

    # -----------------------------------------------------------------
    # Unstructured Dataset Parsers
    # -----------------------------------------------------------------

    def _parse_hospital_information_record(self, raw_text: str) -> dict[str, str] | None:
        """Parse a single hospital information record block into a field dict.

        Args:
            raw_text: Raw text for one record, delimited elsewhere by
                ``"===...==="`` separator lines.

        Returns:
            A dictionary of extracted fields, or ``None`` if the record is
            malformed (missing identifier or required fields).
        """
        blocks = self._split_text_blocks(raw_text)
        if not blocks:
            return None

        record_id = self._extract_record_id(blocks[0])
        if not record_id:
            logger.warning(
                "Hospital information record missing a valid identifier: %s",
                blocks[0][:80],
            )
            return None

        fields: dict[str, str] = {"record_id": record_id}
        for block in blocks[1:]:
            parsed_field = self._extract_labeled_field(block)
            if parsed_field is None:
                continue
            label, value = parsed_field
            fields[label] = value

        if not fields.get("title") or not fields.get("description"):
            logger.warning(
                "Hospital information record %s is missing required fields.",
                record_id,
            )
            return None

        return fields

    def _parse_hospital_information_dataset(self) -> list[Document]:
        """Parse the hospital information dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid
            hospital information record in
            ``knowledge_base/unstructured/hospital_information.txt``.
        """

        source_file = "hospital_information.txt"
        file_path = self.unstructured_dir / source_file
        logger.info("Loading hospital information dataset: %s", file_path)

        content = self._load_text_file(file_path)
        raw_records = [
            chunk.strip()
            for chunk in re.split(r"^=+$", content, flags=re.MULTILINE)
            if chunk.strip()
        ]

        documents: list[Document] = []
        skipped_count = 0

        for index, raw_record in enumerate(raw_records):
            try:
                record = self._parse_hospital_information_record(raw_record)
                if record is None:
                    skipped_count += 1
                    continue

                page_content = self._format_hospital_information_document(record)
                metadata = {
                    "source": "hospital_information",
                    "record_type": "hospital_information",
                    "source_file": source_file,
                    "record_id": record.get("record_id"),
                    "category": record.get("category"),
                    "title": record.get("title"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid hospital information record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid hospital information records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Hospital information parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(raw_records),
            skipped_count,
        )
        return documents

    def _parse_faq_record(self, raw_text: str) -> dict[str, str] | None:
        """Parse a single FAQ record block into a field dict.

        Args:
            raw_text: Raw text for one record, delimited elsewhere by
                ``"---...---"`` separator lines.

        Returns:
            A dictionary with ``record_id``, ``question``, and ``answer``
            keys, or ``None`` if the record is malformed.
        """
        stripped_text = raw_text.strip()
        if not stripped_text:
            return None

        lines = stripped_text.split("\n")

        record_id = self._extract_record_id(lines[0])
        if not record_id:
            logger.warning(
                "FAQ record missing a valid identifier: %s", lines[0][:80]
            )
            return None

        remaining_text = "\n".join(lines[1:]).strip()
        match = re.match(
            r"^Question:\s*\n(?P<question>.*?)\n"
            r"Answer:\s*\n(?P<answer>.*)\Z",
            remaining_text,
            flags=re.DOTALL,
        )

        if not match:
            logger.warning(
                "FAQ record %s does not match the expected Question/Answer structure.",
                record_id,
            )
            return None

        question = " ".join(match.group("question").split())
        answer = " ".join(match.group("answer").split())

        if not question or not answer:
            logger.warning(
                "FAQ record %s is missing question or answer text.", record_id
            )
            return None

        return {"record_id": record_id, "question": question, "answer": answer}

    def _parse_faq_dataset(self) -> list[Document]:
        """Parse the FAQ dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid FAQ
            record in ``knowledge_base/unstructured/faq.txt``.
        """

        source_file = "faq.txt"
        file_path = self.unstructured_dir / source_file
        logger.info("Loading FAQ dataset: %s", file_path)

        content = self._load_text_file(file_path)
        raw_records = [
            chunk.strip()
            for chunk in re.split(r"^-{10,}$", content, flags=re.MULTILINE)
            if chunk.strip()
        ]

        documents: list[Document] = []
        skipped_count = 0

        for index, raw_record in enumerate(raw_records):
            try:
                record = self._parse_faq_record(raw_record)
                if record is None:
                    skipped_count += 1
                    continue

                page_content = self._format_faq_document(record)
                metadata = {
                    "source": "faq",
                    "record_type": "faq",
                    "source_file": source_file,
                    "record_id": record.get("record_id"),
                    "question": record.get("question"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid FAQ record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid FAQ records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "FAQ parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(raw_records),
            skipped_count,
        )
        return documents

    def _parse_patient_guideline_record(self, raw_text: str) -> dict[str, str] | None:
        """Parse a single patient guideline record block into a field dict.

        Args:
            raw_text: Raw text for one record, delimited elsewhere by
                ``"---...---"`` separator lines.

        Returns:
            A dictionary of extracted fields, or ``None`` if the record is
            malformed (missing identifier or required fields).
        """
        blocks = self._split_text_blocks(raw_text)
        if not blocks:
            return None

        record_id = self._extract_record_id(blocks[0])
        if not record_id:
            logger.warning(
                "Patient guideline record missing a valid identifier: %s",
                blocks[0][:80],
            )
            return None

        fields: dict[str, str] = {"record_id": record_id}
        for block in blocks[1:]:
            parsed_field = self._extract_labeled_field(block)
            if parsed_field is None:
                continue
            label, value = parsed_field
            fields[label] = value

        if not fields.get("title") or not fields.get("guideline"):
            logger.warning(
                "Patient guideline record %s is missing required fields.",
                record_id,
            )
            return None

        return fields

    def _parse_patient_guideline_dataset(self) -> list[Document]:
        """Parse the patient guidelines dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid patient
            guideline record in
            ``knowledge_base/unstructured/patient_guidelines.txt``.
        """

        source_file = "patient_guidelines.txt"
        file_path = self.unstructured_dir / source_file
        logger.info("Loading patient guidelines dataset: %s", file_path)

        content = self._load_text_file(file_path)
        raw_records = [
            chunk.strip()
            for chunk in re.split(r"^-{10,}$", content, flags=re.MULTILINE)
            if chunk.strip()
        ]

        documents: list[Document] = []
        skipped_count = 0

        for index, raw_record in enumerate(raw_records):
            try:
                record = self._parse_patient_guideline_record(raw_record)
                if record is None:
                    skipped_count += 1
                    continue

                page_content = self._format_patient_guideline_document(record)
                metadata = {
                    "source": "patient_guidelines",
                    "record_type": "patient_guideline",
                    "source_file": source_file,
                    "record_id": record.get("record_id"),
                    "title": record.get("title"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid patient guideline record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid patient guideline records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Patient guideline parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(raw_records),
            skipped_count,
        )
        return documents

    def _parse_billing_information_record(self, raw_text: str) -> dict[str, str] | None:
        """Parse a single billing information record block into a field dict.

        Args:
            raw_text: Raw text for one record, delimited elsewhere by
                ``"---...---"`` separator lines.

        Returns:
            A dictionary of extracted fields, or ``None`` if the record is
            malformed (missing identifier or required fields).
        """
        blocks = self._split_text_blocks(raw_text)
        if not blocks:
            return None

        record_id = self._extract_record_id(blocks[0])
        if not record_id:
            logger.warning ( 
                "Billing information record missing a valid identifier: %s",
                blocks[0][:80],
            )
            return None

        fields: dict[str, str] = {"record_id": record_id}
        for block in blocks[1:]:
            parsed_field = self._extract_labeled_field(block)
            if parsed_field is None:
                continue
            label, value = parsed_field
            fields[label] = value

        if not fields.get("title") or not fields.get("information"):
            logger.warning(
                "Billing information record %s is missing required fields.",
                record_id,
            )
            return None

        return fields

    def _parse_billing_information_dataset(self) -> list[Document]:
        """Parse the billing information dataset into LangChain documents.

        Returns:
            A list of LangChain Document objects, one for each valid billing
            information record in
            ``knowledge_base/unstructured/billing_information.txt``.
        """

        source_file = "billing_information.txt"
        file_path = self.unstructured_dir / source_file
        logger.info("Loading billing information dataset: %s", file_path)

        content = self._load_text_file(file_path)
        raw_records = [
            chunk.strip()
            for chunk in re.split(r"^-{10,}$", content, flags=re.MULTILINE)
            if chunk.strip()
        ]

        documents: list[Document] = []
        skipped_count = 0

        for index, raw_record in enumerate(raw_records):
            try:
                record = self._parse_billing_information_record(raw_record)
                if record is None:
                    skipped_count += 1
                    continue

                page_content = self._format_billing_information_document(record)
                metadata = {
                    "source": "billing_information",
                    "record_type": "billing_information",
                    "source_file": source_file,
                    "record_id": record.get("record_id"),
                    "title": record.get("title"),
                }

                document = self._create_document(page_content, metadata)
                documents.append(document)
            except Exception:
                logger.exception(
                    "Skipping invalid billing information record at index %s due to processing failure.",
                    index,
                )
                skipped_count += 1
                continue

        logger.info(
            "Parsed %s valid billing information records from %s.",
            len(documents),
            file_path,
        )
        logger.info(
            "Billing information parsing summary: %s/%s records converted successfully, %s skipped.",
            len(documents),
            len(raw_records),
            skipped_count,
        )
        return documents

    # -----------------------------------------------------------------
    # Public APIs
    # -----------------------------------------------------------------

    def load_all_documents(self) -> list[Document]:
        """Load all available knowledge base documents.

        This is the public entry point for building the in-memory document
        collection used by the RAG pipeline. It loads every structured
        dataset in a fixed order (doctors, departments, symptoms, diseases,
        medicines, navigation, appointments, insurance, emergency protocols),
        followed by every unstructured dataset in a fixed order (hospital
        information, FAQ, patient guidelines, billing information), and
        returns the combined LangChain documents as a single list.

        Returns:
            A list of LangChain Document objects representing the loaded
            knowledge base content.
        """

        logger.info("Starting knowledge base document loading.")

        documents: list[Document] = []
        documents.extend(self._parse_doctor_dataset())
        documents.extend(self._parse_department_dataset())
        documents.extend(self._parse_symptom_dataset())
        documents.extend(self._parse_disease_dataset())
        documents.extend(self._parse_medicine_dataset())
        documents.extend(self._parse_navigation_dataset())
        documents.extend(self._parse_appointment_dataset())
        documents.extend(self._parse_insurance_dataset())
        documents.extend(self._parse_emergency_dataset())
        documents.extend(self._parse_hospital_information_dataset())
        documents.extend(self._parse_faq_dataset())
        documents.extend(self._parse_patient_guideline_dataset())
        documents.extend(self._parse_billing_information_dataset())

        logger.info(
            "Completed knowledge base document loading with %s documents.",
            len(documents),
        )
        return documents