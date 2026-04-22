def format_output(predictions):
    if not predictions:
        return "No strong emotion detected"
    return ", ".join(predictions)
