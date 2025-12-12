package com.visionvet.ai.feature.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.visionvet.ai.core.database.model.Analysis
import com.visionvet.ai.core.database.model.ParasiteType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.*

class DashboardViewModel : ViewModel() {

    private val _pendingUploads = MutableStateFlow(0)
    val pendingUploads: StateFlow<Int> = _pendingUploads.asStateFlow()

    private val _selectedTimeFrame = MutableStateFlow(TimeFrame.WEEK)
    val selectedTimeFrame: StateFlow<TimeFrame> = _selectedTimeFrame.asStateFlow()

    private val _analysisSummary = MutableStateFlow<AnalysisSummary?>(null)
    val analysisSummary: StateFlow<AnalysisSummary?> = _analysisSummary.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _showError = MutableStateFlow(false)
    val showError: StateFlow<Boolean> = _showError.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    enum class TimeFrame(val displayName: String) {
        DAY("Day"),
        WEEK("Week"),
        MONTH("Month"),
        YEAR("Year");
    }

    data class AnalysisSummary(
        val totalCount: Int,
        val parasiteCounts: Map<ParasiteType, Int>,
        val dateRange: Pair<Date, Date>
    )

    fun updateTimeFrame(timeFrame: TimeFrame) {
        _selectedTimeFrame.value = timeFrame
        generateSummary()
    }

    fun loadPendingUploads() {
        viewModelScope.launch {
            try {
                // TODO: Repository'den gerçek veri çek
                // Şimdilik mock data
                _pendingUploads.value = 2
            } catch (e: Exception) {
                _errorMessage.value = "Failed to count pending uploads: ${e.localizedMessage}"
                _showError.value = true
            }
        }
    }

    fun uploadPendingAnalyses() {
        viewModelScope.launch {
            try {
                _isLoading.value = true

                // TODO: Repository'den pending analyses çek ve upload et
                // Şimdilik mock upload process

                // Mock upload delay
                kotlinx.coroutines.delay(2000)

                _pendingUploads.value = 0
                loadPendingUploads()

            } catch (e: Exception) {
                _errorMessage.value = "Failed to upload analyses: ${e.localizedMessage}"
                _showError.value = true
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun generateSummary() {
        viewModelScope.launch {
            try {
                val now = Date()
                val startDate = when (_selectedTimeFrame.value) {
                    TimeFrame.DAY -> Calendar.getInstance().apply {
                        time = now
                        add(Calendar.DAY_OF_YEAR, -1)
                    }.time
                    TimeFrame.WEEK -> Calendar.getInstance().apply {
                        time = now
                        add(Calendar.DAY_OF_YEAR, -7)
                    }.time
                    TimeFrame.MONTH -> Calendar.getInstance().apply {
                        time = now
                        add(Calendar.MONTH, -1)
                    }.time
                    TimeFrame.YEAR -> Calendar.getInstance().apply {
                        time = now
                        add(Calendar.YEAR, -1)
                    }.time
                }

                // TODO: Repository'den gerçek veri çek
                // Şimdilik mock data
                val mockParasiteCounts: Map<ParasiteType, Int> = mapOf(
                    ParasiteType.ASCARIS to 5,
                    ParasiteType.HOOKWORM to 3,
                    ParasiteType.TRICHURIS to 2,
                    ParasiteType.HYMENOLEPIS to 1
                )

                _analysisSummary.value = AnalysisSummary(
                    totalCount = mockParasiteCounts.values.sum(),
                    parasiteCounts = mockParasiteCounts,
                    dateRange = Pair(startDate, now)
                )

            } catch (e: Exception) {
                _errorMessage.value = "Failed to generate summary: ${e.localizedMessage}"
                _showError.value = true
            }
        }
    }

    fun dismissError() {
        _showError.value = false
        _errorMessage.value = null
    }

    fun deleteAnalysis(analysisId: String) {
        viewModelScope.launch {
            try {
                // TODO: Repository'den analysis sil

                // Refresh data
                loadPendingUploads()
                generateSummary()

            } catch (e: Exception) {
                _errorMessage.value = "Failed to delete analysis: ${e.localizedMessage}"
                _showError.value = true
            }
        }
    }

    init {
        loadPendingUploads()
        generateSummary()
    }
}
