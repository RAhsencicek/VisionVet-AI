package com.visionvet.ai.core.database.model

enum class ParasiteType(val displayName: String) {
    ASCARIS("Ascaris"),
    HOOKWORM("Hookworm"),
    TRICHURIS("Trichuris"),
    HYMENOLEPIS("Hymenolepis");

    companion object {
        fun fromString(value: String): ParasiteType? {
            return entries.find { it.displayName.equals(value, ignoreCase = true) }
        }
    }
}
