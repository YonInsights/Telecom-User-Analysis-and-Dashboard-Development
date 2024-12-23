def compute_satisfaction_score(engagement_scores, experience_scores):
    """
    Compute satisfaction score as the average of engagement and experience scores.

    Args:
        engagement_scores (pd.Series): Engagement scores.
        experience_scores (pd.Series): Experience scores.

    Returns:
        pd.Series: Satisfaction scores.
    """
    return (engagement_scores + experience_scores) / 2
