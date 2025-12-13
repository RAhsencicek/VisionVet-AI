package com.visionvet.ai.feature.dashboard

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.visionvet.ai.ui.components.*
import com.visionvet.ai.ui.theme.*
import java.text.SimpleDateFormat
import java.util.*

/**
 * Modern Dashboard Screen
 * Features: Hero scan card, animated stats, recent analyses
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    onNavigateToScan: () -> Unit = {},
    onNavigateToHistory: () -> Unit = {}
) {
    // Mock data for demo
    val totalAnalyses = remember { 23 }
    val todayAnalyses = remember { 3 }
    val successRate = remember { 94.5f }
    
    val recentAnalyses = remember {
        listOf(
            RecentAnalysis("Escherichia coli", 94.5f, "2 dakika önce", true),
            RecentAnalysis("Staphylococcus aureus", 87.2f, "1 saat önce", true),
            RecentAnalysis("Bilinmeyen örnek", 0f, "3 saat önce", false)
        )
    }
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // Header
        item {
            DashboardHeader()
        }
        
        // Hero Scan Card
        item {
            HeroScanCard(
                onClick = onNavigateToScan
            )
        }
        
        // Stats Grid
        item {
            StatsGrid(
                totalAnalyses = totalAnalyses,
                todayAnalyses = todayAnalyses,
                successRate = successRate
            )
        }
        
        // Recent Analyses Section
        item {
            RecentAnalysesHeader(
                onSeeAllClick = onNavigateToHistory
            )
        }
        
        // Recent Analysis Cards
        items(recentAnalyses) { analysis ->
            RecentAnalysisCard(analysis = analysis)
        }
        
        // Bottom spacing for navigation
        item {
            Spacer(modifier = Modifier.height(80.dp))
        }
    }
}

@Composable
private fun DashboardHeader() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = "Merhaba 👋",
                style = MaterialTheme.typography.bodyLarge,
                color = TextSecondary
            )
            Text(
                text = "VisionVet AI",
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontWeight = FontWeight.Bold
                ),
                color = Color.White
            )
        }
        
        // Notification bell
        Box(
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(14.dp))
                .background(DarkCard),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Outlined.Notifications,
                contentDescription = "Notifications",
                tint = TextSecondary
            )
        }
    }
}

@Composable
private fun HeroScanCard(
    onClick: () -> Unit
) {
    val infiniteTransition = rememberInfiniteTransition(label = "hero")
    
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.6f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = EaseInOut),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
    )
    
    GradientCard(
        modifier = Modifier
            .fillMaxWidth()
            .height(180.dp),
        gradientColors = listOf(
            BacteriaBlue,
            DeepBlue,
            ElectricPurple.copy(alpha = 0.8f)
        ),
        onClick = onClick
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = "Yeni Analiz",
                    style = MaterialTheme.typography.headlineSmall.copy(
                        fontWeight = FontWeight.Bold
                    ),
                    color = Color.White
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Text(
                    text = "Bakteriyel koloni görüntüsü\ntarayın ve anında sonuç alın",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White.copy(alpha = 0.8f)
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.CameraAlt,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                    Text(
                        text = "Taramaya Başla",
                        style = MaterialTheme.typography.labelLarge,
                        color = Color.White
                    )
                    Icon(
                        imageVector = Icons.Filled.ArrowForward,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }
            
            // Animated bacteria icon
            Box(
                modifier = Modifier
                    .size(100.dp)
                    .background(
                        color = Color.White.copy(alpha = 0.1f),
                        shape = RoundedCornerShape(20.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Biotech,
                    contentDescription = null,
                    tint = Color.White.copy(alpha = glowAlpha + 0.4f),
                    modifier = Modifier.size(60.dp)
                )
            }
        }
    }
}

@Composable
private fun StatsGrid(
    totalAnalyses: Int,
    todayAnalyses: Int,
    successRate: Float
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        StatCard(
            title = "Toplam",
            value = totalAnalyses.toString(),
            icon = Icons.Outlined.Assessment,
            iconTint = BacteriaBlue,
            modifier = Modifier.weight(1f)
        )
        
        StatCard(
            title = "Bugün",
            value = todayAnalyses.toString(),
            icon = Icons.Outlined.Today,
            iconTint = MicrobeGreen,
            modifier = Modifier.weight(1f)
        )
        
        StatCard(
            title = "Başarı",
            value = "${successRate.toInt()}%",
            icon = Icons.Outlined.TrendingUp,
            iconTint = ElectricPurple,
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun RecentAnalysesHeader(
    onSeeAllClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "Son Analizler",
            style = MaterialTheme.typography.titleMedium.copy(
                fontWeight = FontWeight.SemiBold
            ),
            color = Color.White
        )
        
        TextButton(onClick = onSeeAllClick) {
            Text(
                text = "Tümünü Gör",
                style = MaterialTheme.typography.labelLarge,
                color = BacteriaBlue
            )
            Icon(
                imageVector = Icons.Filled.ChevronRight,
                contentDescription = null,
                tint = BacteriaBlue,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@Composable
private fun RecentAnalysisCard(
    analysis: RecentAnalysis
) {
    GlassmorphicCard(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 16.dp
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Status indicator
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .background(
                            color = if (analysis.isSuccess) 
                                MicrobeGreen.copy(alpha = 0.15f) 
                            else 
                                AlertRed.copy(alpha = 0.15f),
                            shape = RoundedCornerShape(12.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (analysis.isSuccess) 
                            Icons.Filled.CheckCircle 
                        else 
                            Icons.Filled.Error,
                        contentDescription = null,
                        tint = if (analysis.isSuccess) MicrobeGreen else AlertRed,
                        modifier = Modifier.size(24.dp)
                    )
                }
                
                Column {
                    Text(
                        text = analysis.bacteriaName,
                        style = MaterialTheme.typography.bodyLarge.copy(
                            fontWeight = FontWeight.Medium
                        ),
                        color = Color.White
                    )
                    Text(
                        text = analysis.timeAgo,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
            }
            
            if (analysis.isSuccess && analysis.confidence > 0) {
                // Confidence badge
                Box(
                    modifier = Modifier
                        .background(
                            color = BacteriaBlue.copy(alpha = 0.15f),
                            shape = RoundedCornerShape(8.dp)
                        )
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = "${analysis.confidence.toInt()}%",
                        style = MaterialTheme.typography.labelMedium.copy(
                            fontWeight = FontWeight.SemiBold
                        ),
                        color = BacteriaBlue
                    )
                }
            }
        }
    }
}

// Data class for recent analysis
private data class RecentAnalysis(
    val bacteriaName: String,
    val confidence: Float,
    val timeAgo: String,
    val isSuccess: Boolean
)
