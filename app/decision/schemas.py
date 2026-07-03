from enum import StrEnum


class Channel(StrEnum):
    EMAIL = "email"
    SMS = "sms"
    ONSITE_BANNER = "onsite_banner"


class GoalMetric(StrEnum):
    INFLOW_RATE = "inflow_rate"
    BOOKING_CONVERSION_RATE = "booking_conversion_rate"
    FUNNEL_STEP_RATE = "funnel_step_rate"


class GoalBasis(StrEnum):
    PROMOTION_AVERAGE = "promotion_average"
    ALL_SEGMENTS = "all_segments"


class PromotionRunStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    EVALUATING = "evaluating"
    PARTIAL_GOAL_MET = "partial_goal_met"
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    INSUFFICIENT_DATA = "insufficient_data"
    STOPPED = "stopped"


class AdExperimentStatus(StrEnum):
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    EVALUATING = "evaluating"
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    INSUFFICIENT_DATA = "insufficient_data"
    STOPPED = "stopped"


class PromotionEvaluationStatus(StrEnum):
    GOAL_MET = "goal_met"
    GOAL_NOT_MET = "goal_not_met"
    PARTIAL_GOAL_MET = "partial_goal_met"
    INSUFFICIENT_DATA = "insufficient_data"


class AssignmentSource(StrEnum):
    DECISION_BATCH = "decision_batch"
    FALLBACK = "fallback"
    MANUAL = "manual"
    FIXTURE = "fixture"
