package com.visionvet.ai.core.database.model

enum class AnalysisType(val displayName: String) {
    PARASITE("Parasite Analysis"),
    BLOOD("Blood Analysis"),
    URINE("Urine Analysis");

    companion object {
        fun fromString(value: String): AnalysisType? {
            return entries.find { it.displayName.equals(value, ignoreCase = true) }
        }
    }
}
