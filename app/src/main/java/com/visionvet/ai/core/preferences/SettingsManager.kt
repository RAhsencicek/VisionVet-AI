package com.visionvet.ai.core.preferences

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Settings Manager using DataStore
 * Handles app preferences persistence
 */
class SettingsManager(private val context: Context) {
    
    companion object {
        private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "visionvet_settings")
        
        // Keys
        private val DARK_MODE = booleanPreferencesKey("dark_mode")
        private val NOTIFICATIONS = booleanPreferencesKey("notifications_enabled")
        private val AUTO_SAVE = booleanPreferencesKey("auto_save_results")
        private val LANGUAGE = stringPreferencesKey("language")
        private val CONFIDENCE_THRESHOLD = stringPreferencesKey("confidence_threshold")
        
        @Volatile
        private var INSTANCE: SettingsManager? = null
        
        fun getInstance(context: Context): SettingsManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: SettingsManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    // Dark Mode
    val isDarkModeEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences -> preferences[DARK_MODE] ?: true }
    
    suspend fun setDarkMode(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[DARK_MODE] = enabled
        }
    }
    
    // Notifications
    val isNotificationsEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences -> preferences[NOTIFICATIONS] ?: true }
    
    suspend fun setNotifications(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[NOTIFICATIONS] = enabled
        }
    }
    
    // Auto Save
    val isAutoSaveEnabled: Flow<Boolean> = context.dataStore.data
        .map { preferences -> preferences[AUTO_SAVE] ?: true }
    
    suspend fun setAutoSave(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[AUTO_SAVE] = enabled
        }
    }
    
    // Language
    val language: Flow<String> = context.dataStore.data
        .map { preferences -> preferences[LANGUAGE] ?: "tr" }
    
    suspend fun setLanguage(lang: String) {
        context.dataStore.edit { preferences ->
            preferences[LANGUAGE] = lang
        }
    }
    
    // Confidence Threshold
    val confidenceThreshold: Flow<String> = context.dataStore.data
        .map { preferences -> preferences[CONFIDENCE_THRESHOLD] ?: "70" }
    
    suspend fun setConfidenceThreshold(threshold: String) {
        context.dataStore.edit { preferences ->
            preferences[CONFIDENCE_THRESHOLD] = threshold
        }
    }
    
    // Clear all settings
    suspend fun clearAll() {
        context.dataStore.edit { preferences ->
            preferences.clear()
        }
    }
}
