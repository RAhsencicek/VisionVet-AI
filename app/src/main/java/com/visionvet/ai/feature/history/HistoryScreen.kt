package com.visionvet.ai.feature.history

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.core.database.model.Analysis
import com.visionvet.ai.core.database.model.ParasiteResult
import com.visionvet.ai.core.database.model.ParasiteType
import com.visionvet.ai.ui.theme.VisionVetAITheme
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    analyses: List<Analysis> = emptyList(),
    onAnalysisClick: (Analysis) -> Unit = {}
) {
    var searchText by remember { mutableStateOf("") }
    var showFilterSheet by remember { mutableStateOf(false) }
    var filterByParasite by remember { mutableStateOf<ParasiteType?>(null) }
    var filterByUploadStatus by remember { mutableStateOf<Boolean?>(null) }

    val filteredAnalyses = remember(analyses, searchText, filterByParasite, filterByUploadStatus) {
        var result = analyses

        // Apply search filter
        if (searchText.isNotEmpty()) {
            result = result.filter { analysis ->
                val locationMatch = analysis.location.contains(searchText, ignoreCase = true)
                val notesMatch = analysis.notes.contains(searchText, ignoreCase = true)
                val parasiteMatch = analysis.results.any {
                    it.type.displayName.contains(searchText, ignoreCase = true)
                }
                locationMatch || notesMatch || parasiteMatch
            }
        }

        // Apply parasite filter
        filterByParasite?.let { filterParasite ->
            result = result.filter { analysis ->
                analysis.results.any { it.type == filterParasite }
            }
        }

        // Apply upload status filter
        filterByUploadStatus?.let { uploadStatus ->
            result = result.filter { it.isUploaded == uploadStatus }
        }

        result.sortedByDescending { it.timestamp }
    }

    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Top App Bar with Search
        TopAppBar(
            title = { Text("Analysis History") },
            actions = {
                IconButton(onClick = { showFilterSheet = true }) {
                    Icon(
                        Icons.Default.Menu,
                        contentDescription = "Filter",
                        tint = if (filterByParasite != null || filterByUploadStatus != null) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurface
                        }
                    )
                }
            }
        )

        // Search Bar
        OutlinedTextField(
            value = searchText,
            onValueChange = { searchText = it },
            placeholder = { Text("Search analyses...") },
            leadingIcon = {
                Icon(Icons.Default.Search, contentDescription = null)
            },
            trailingIcon = {
                if (searchText.isNotEmpty()) {
                    IconButton(onClick = { searchText = "" }) {
                        Icon(Icons.Default.Clear, contentDescription = "Clear")
                    }
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            shape = RoundedCornerShape(12.dp)
        )

        // Results
        when {
            filteredAnalyses.isEmpty() && analyses.isEmpty() -> {
                EmptyStateView()
            }
            filteredAnalyses.isEmpty() -> {
                NoResultsView(onClearFilters = {
                    searchText = ""
                    filterByParasite = null
                    filterByUploadStatus = null
                })
            }
            else -> {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(filteredAnalyses) { analysis ->
                        AnalysisHistoryCard(
                            analysis = analysis,
                            onClick = { onAnalysisClick(analysis) }
                        )
                    }
                }
            }
        }
    }

    // Filter Bottom Sheet
    if (showFilterSheet) {
        ModalBottomSheet(
            onDismissRequest = { showFilterSheet = false }
        ) {
            FilterBottomSheet(
                filterByParasite = filterByParasite,
                filterByUploadStatus = filterByUploadStatus,
                onParasiteFilterChange = { filterByParasite = it },
                onUploadStatusFilterChange = { filterByUploadStatus = it },
                onResetFilters = {
                    filterByParasite = null
                    filterByUploadStatus = null
                },
                onDismiss = { showFilterSheet = false }
            )
        }
    }
}

@Composable
private fun AnalysisHistoryCard(
    analysis: Analysis,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                // Location and date
                Text(
                    text = analysis.location,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface
                )

                Text(
                    text = analysis.formattedDate,
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                // Dominant parasite
                analysis.dominantParasite?.let { parasite ->
                    Row(
                        modifier = Modifier.padding(top = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(
                                    Color(0xFF34C759),
                                    shape = androidx.compose.foundation.shape.CircleShape
                                )
                        )
                        Text(
                            text = parasite.displayName,
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(start = 6.dp)
                        )
                    }
                }

                // Upload status
                Row(
                    modifier = Modifier.padding(top = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = if (analysis.isUploaded) Icons.Default.CloudDone else Icons.Default.CloudQueue,
                        contentDescription = null,
                        tint = if (analysis.isUploaded) Color(0xFF34C759) else Color(0xFFFF9500),
                        modifier = Modifier.size(16.dp)
                    )
                    Text(
                        text = if (analysis.isUploaded) "Uploaded" else "Pending",
                        fontSize = 12.sp,
                        color = if (analysis.isUploaded) Color(0xFF34C759) else Color(0xFFFF9500),
                        modifier = Modifier.padding(start = 4.dp)
                    )
                }
            }

            Icon(
                Icons.AutoMirrored.Filled.ArrowForward,
                contentDescription = "View Details",
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun EmptyStateView() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                Icons.Default.History,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "No analyses yet",
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Start by taking your first analysis",
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun NoResultsView(
    onClearFilters: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                Icons.Default.SearchOff,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "No results found",
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            TextButton(onClick = onClearFilters) {
                Text("Clear filters")
            }
        }
    }
}

@Composable
private fun FilterBottomSheet(
    filterByParasite: ParasiteType?,
    filterByUploadStatus: Boolean?,
    onParasiteFilterChange: (ParasiteType?) -> Unit,
    onUploadStatusFilterChange: (Boolean?) -> Unit,
    onResetFilters: () -> Unit,
    onDismiss: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        Text(
            text = "Filter Options",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.padding(bottom = 24.dp)
        )

        // Filter by Parasite
        Text(
            text = "Filter by Parasite",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        // All parasites option
        FilterOption(
            text = "All",
            isSelected = filterByParasite == null,
            onClick = { onParasiteFilterChange(null) }
        )

        ParasiteType.entries.forEach { type ->
            FilterOption(
                text = type.displayName,
                isSelected = filterByParasite == type,
                onClick = { onParasiteFilterChange(type) },
                leadingIcon = getParasiteIcon(type),
                iconTint = getParasiteColor(type)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Filter by Upload Status
        Text(
            text = "Filter by Upload Status",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        FilterOption(
            text = "All",
            isSelected = filterByUploadStatus == null,
            onClick = { onUploadStatusFilterChange(null) }
        )

        FilterOption(
            text = "Uploaded",
            isSelected = filterByUploadStatus == true,
            onClick = { onUploadStatusFilterChange(true) },
            leadingIcon = Icons.Default.CheckCircle,
            iconTint = Color(0xFF34C759)
        )

        FilterOption(
            text = "Not Uploaded",
            isSelected = filterByUploadStatus == false,
            onClick = { onUploadStatusFilterChange(false) },
            leadingIcon = Icons.Default.Warning,
            iconTint = Color(0xFFFF9500)
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Action buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedButton(
                onClick = onResetFilters,
                modifier = Modifier.weight(1f)
            ) {
                Text("Reset Filters")
            }

            Button(
                onClick = onDismiss,
                modifier = Modifier.weight(1f)
            ) {
                Text("Done")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
private fun FilterOption(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    leadingIcon: androidx.compose.ui.graphics.vector.ImageVector? = null,
    iconTint: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        leadingIcon?.let { icon ->
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = iconTint,
                modifier = Modifier.size(20.dp)
            )
        }

        Text(
            text = text,
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.weight(1f)
        )

        if (isSelected) {
            Icon(
                Icons.Default.Check,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary
            )
        }
    }
}

// Helper functions for parasite display
private fun getParasiteIcon(type: ParasiteType): androidx.compose.ui.graphics.vector.ImageVector {
    return when (type) {
        ParasiteType.ASCARIS -> Icons.Default.BugReport
        ParasiteType.HOOKWORM -> Icons.Default.Circle
        ParasiteType.TRICHURIS -> Icons.Default.Warning
        ParasiteType.HYMENOLEPIS -> Icons.Default.Science
    }
}

private fun getParasiteColor(type: ParasiteType): Color {
    return when (type) {
        ParasiteType.ASCARIS -> Color(0xFF007AFF) // Blue
        ParasiteType.HOOKWORM -> Color(0xFF34C759) // Green
        ParasiteType.TRICHURIS -> Color(0xFFFF9500) // Orange
        ParasiteType.HYMENOLEPIS -> Color(0xFFAF52DE) // Purple
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun HistoryScreenPreview() {
    VisionVetAITheme {
        HistoryScreen(
            analyses = getSampleAnalyses(),
            onAnalysisClick = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun HistoryScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        HistoryScreen(
            analyses = getSampleAnalyses(),
            onAnalysisClick = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun HistoryScreenEmptyPreview() {
    VisionVetAITheme {
        HistoryScreen(
            analyses = emptyList(),
            onAnalysisClick = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun AnalysisHistoryCardPreview() {
    VisionVetAITheme {
        AnalysisHistoryCard(
            analysis = getSampleAnalyses().first(),
            onClick = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun EmptyStateViewPreview() {
    VisionVetAITheme {
        EmptyStateView()
    }
}

@Preview(showBackground = true)
@Composable
fun NoResultsViewPreview() {
    VisionVetAITheme {
        NoResultsView(onClearFilters = {})
    }
}

// Sample data for previews
private fun getSampleAnalyses(): List<Analysis> {
    val currentTime = System.currentTimeMillis()
    return listOf(
        Analysis(
            id = "1",
            userId = "user1",
            location = "Lab Room 1",
            timestamp = currentTime - 86400000, // 1 day ago
            notes = "Sample from patient A",
            results = listOf(
                ParasiteResult(ParasiteType.ASCARIS, 0.85, currentTime - 86400000),
                ParasiteResult(ParasiteType.HOOKWORM, 0.15, currentTime - 86400000)
            ),
            isUploaded = true
        ),
        Analysis(
            id = "2",
            userId = "user1",
            location = "Field Study Site B",
            timestamp = currentTime - 172800000, // 2 days ago
            notes = "Routine check sample",
            results = listOf(
                ParasiteResult(ParasiteType.TRICHURIS, 0.92, currentTime - 172800000)
            ),
            isUploaded = false
        ),
        Analysis(
            id = "3",
            userId = "user1",
            location = "Veterinary Clinic",
            timestamp = currentTime - 259200000, // 3 days ago
            notes = "Emergency case analysis",
            results = listOf(
                ParasiteResult(ParasiteType.HYMENOLEPIS, 0.75, currentTime - 259200000),
                ParasiteResult(ParasiteType.ASCARIS, 0.25, currentTime - 259200000)
            ),
            isUploaded = true
        )
    )
}
