def validate_tech_pack(data: dict) -> tuple[bool, list[str]]:
    """
    Validate the generated tech pack.

    Returns:
        (is_valid, messages)

    Hard errors prevent the tech pack from being accepted.
    Warnings are included in the messages but do not fail validation.
    """

    errors = []
    warnings = []

    # ============================================================
    # 1. ROOT STRUCTURE
    # ============================================================

    required_fields = [
        "product_name",
        "product_description",
        "design_evidence",
        "cad_vector",
        "materials_bom",
        "measurements_cm",
        "construction_details",
        "colorways",
        "quality_control",
    ]

    for field in required_fields:
        if field not in data:
            errors.append(
                f"Missing required top-level field: '{field}'"
            )

    # Stop here if the basic structure is missing.
    if errors:
        return False, errors

    # ============================================================
    # 2. PRODUCT DESCRIPTION
    # ============================================================

    product = data["product_description"]

    if not isinstance(product, dict):
        errors.append(
            "'product_description' must be an object."
        )
    else:
        required_product_fields = [
            "category",
            "product_type",
            "intended_use",
            "design_summary",
        ]

        for field in required_product_fields:
            if not product.get(field):
                errors.append(
                    f"Missing product_description field: '{field}'"
                )

    # ============================================================
    # 3. DESIGN EVIDENCE
    # ============================================================

    evidence = data["design_evidence"]

    if not isinstance(evidence, dict):
        errors.append(
            "'design_evidence' must be an object."
        )
    else:

        observed = evidence.get("observed_features")

        if not isinstance(observed, list):
            errors.append(
                "'observed_features' must be a list."
            )
        elif len(observed) == 0:
            warnings.append(
                "No explicit observed visual features were recorded."
            )

        for field in [
            "inferred_features",
            "assumptions",
            "uncertainties",
            "conflicts_with_user_description",
        ]:
            if field in evidence and not isinstance(
                evidence[field], list
            ):
                errors.append(
                    f"design_evidence['{field}'] must be a list."
                )

    # ============================================================
    # 4. CAD VECTOR VALIDATION
    # ============================================================

    cad = data["cad_vector"]

    if not isinstance(cad, dict):
        errors.append("'cad_vector' must be an object.")
    else:

        bounds = cad.get("canvas_bounds")

        if (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or not all(
                isinstance(value, (int, float))
                for value in bounds
            )
        ):
            errors.append(
                "CAD canvas_bounds must be "
                "[xmin, xmax, ymin, ymax]."
            )

        else:

            xmin, xmax, ymin, ymax = bounds

            if xmin >= xmax or ymin >= ymax:
                errors.append(
                    f"Invalid CAD coordinate boundaries: {bounds}"
                )

            primitives = cad.get("primitives", [])

            if not isinstance(primitives, list):
                errors.append(
                    "CAD primitives must be a list."
                )
                primitives = []

            if len(primitives) == 0:
                errors.append(
                    "CAD contains no primitives."
                )

            for idx, primitive in enumerate(primitives):

                if not isinstance(primitive, dict):
                    errors.append(
                        f"CAD primitive #{idx} must be an object."
                    )
                    continue

                points = primitive.get("points", [])
                role = primitive.get("role", "")
                linestyle = primitive.get(
                    "linestyle",
                    "-"
                )

                # ------------------------------------------------
                # Topstitching rule
                # ------------------------------------------------

                if (
                    role == "topstitching"
                    and linestyle != "--"
                ):
                    errors.append(
                        f"CAD primitive #{idx}: "
                        "topstitching must use dashed "
                        "linestyle '--'."
                    )

                # ------------------------------------------------
                # Point validation
                # ------------------------------------------------

                if not isinstance(points, list):
                    errors.append(
                        f"CAD primitive #{idx}: "
                        "'points' must be a list."
                    )
                    continue

                for point in points:

                    if (
                        not isinstance(point, (list, tuple))
                        or len(point) < 2
                    ):
                        errors.append(
                            f"CAD primitive #{idx}: "
                            f"invalid point format: {point}"
                        )
                        continue

                    x, y = point[0], point[1]

                    if not isinstance(x, (int, float)):
                        errors.append(
                            f"CAD primitive #{idx}: "
                            f"X coordinate '{x}' is not numeric."
                        )
                        continue

                    if not isinstance(y, (int, float)):
                        errors.append(
                            f"CAD primitive #{idx}: "
                            f"Y coordinate '{y}' is not numeric."
                        )
                        continue

                    if not (xmin <= x <= xmax):
                        errors.append(
                            f"CAD primitive #{idx}: "
                            f"X={x} outside CAD bounds."
                        )

                    if not (ymin <= y <= ymax):
                        errors.append(
                            f"CAD primitive #{idx}: "
                            f"Y={y} outside CAD bounds."
                        )

    # ============================================================
    # 5. BOM VALIDATION
    # ============================================================

    bom = data["materials_bom"]

    if not isinstance(bom, list):
        errors.append(
            "'materials_bom' must be a list."
        )

    elif len(bom) == 0:
        errors.append(
            "Bill of Materials (BOM) cannot be empty."
        )

    else:

        for idx, item in enumerate(bom):

            if not isinstance(item, dict):
                errors.append(
                    f"BOM item #{idx} must be an object."
                )
                continue

            if not item.get("item"):
                errors.append(
                    f"BOM item #{idx} is missing 'item'."
                )

            if not item.get("description"):
                errors.append(
                    f"BOM item #{idx} is missing "
                    "'description'."
                )

            status = item.get("material_status")

            if status not in [
                "observed",
                "inferred",
                "provisional",
            ]:
                warnings.append(
                    f"BOM item #{idx} has no valid "
                    "material_status."
                )

    # ============================================================
    # 6. MEASUREMENT / SIZE CHART VALIDATION
    # ============================================================

    measurements = data["measurements_cm"]

    if not isinstance(measurements, dict):
        errors.append(
            "'measurements_cm' must be an object."
        )

    else:

        sizes = measurements.get("sizes", {})
        size_system = str(
            measurements.get(
                "size_system",
                ""
            )
        ).upper()

        if not isinstance(sizes, dict):
            errors.append(
                "measurements_cm['sizes'] must be an object."
            )

        elif len(sizes) == 0:
            errors.append(
                "Measurement size chart cannot be empty."
            )

        elif size_system != "OS" and len(sizes) < 3:
            errors.append(
                "Sized product requires at least "
                "3 size columns "
                f"(found {len(sizes)})."
            )

        # Physical measurements cannot be recovered from
        # an image without a known scale.
        measurement_status = measurements.get(
            "measurement_status"
        )

        if measurement_status != "provisional":
            warnings.append(
                "Measurement status is not marked "
                "'provisional'. Image-based dimensions "
                "should not be treated as physically measured."
            )

        points_of_measure = measurements.get(
            "points_of_measure",
            []
        )

        if not isinstance(points_of_measure, list):
            errors.append(
                "points_of_measure must be a list."
            )

        elif len(points_of_measure) == 0:
            warnings.append(
                "No measurement points of measure "
                "were specified."
            )

    # ============================================================
    # 7. CONSTRUCTION DETAILS
    # ============================================================

    construction = data["construction_details"]

    if not isinstance(construction, list):
        errors.append(
            "'construction_details' must be a list."
        )

    elif len(construction) == 0:
        errors.append(
            "Construction details cannot be empty."
        )

    # ============================================================
    # 8. COLORWAYS
    # ============================================================

    colorways = data["colorways"]

    if not isinstance(colorways, list):
        errors.append(
            "'colorways' must be a list."
        )

    elif len(colorways) == 0:
        errors.append(
            "Colorway breakdown cannot be empty."
        )

    else:

        for idx, colorway in enumerate(colorways):

            if not isinstance(colorway, dict):
                errors.append(
                    f"Colorway #{idx} must be an object."
                )
                continue

            if not colorway.get("name"):
                errors.append(
                    f"Colorway #{idx} missing 'name'."
                )

            if not colorway.get("main_color"):
                warnings.append(
                    f"Colorway #{idx} has no main_color."
                )

            if "construction_notes" in colorway:
                if not isinstance(
                    colorway["construction_notes"],
                    list
                ):
                    errors.append(
                        f"Colorway #{idx}: "
                        "'construction_notes' must be a list."
                    )

    # ============================================================
    # 9. QUALITY CONTROL
    # ============================================================

    qc = data["quality_control"]

    if not isinstance(qc, dict):
        errors.append(
            "'quality_control' must be an object."
        )

    else:

        if "warnings" in qc and not isinstance(
            qc["warnings"],
            list
        ):
            errors.append(
                "quality_control['warnings'] "
                "must be a list."
            )

        if "missing_information" in qc and not isinstance(
            qc["missing_information"],
            list
        ):
            errors.append(
                "quality_control['missing_information'] "
                "must be a list."
            )

    # ============================================================
    # 10. GENERAL WARNINGS
    # ============================================================

    if not evidence.get("assumptions"):
        warnings.append(
            "No explicit manufacturing assumptions "
            "were recorded."
        )

    if not evidence.get("uncertainties"):
        warnings.append(
            "No visual uncertainties were recorded."
        )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    messages = errors + warnings

    return len(errors) == 0, messages
