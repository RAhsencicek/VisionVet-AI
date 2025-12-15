package com.visionvet.ai.feature.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.visionvet.ai.core.preferences.SettingsManager
import com.visionvet.ai.ui.components.GlassmorphicCard
import com.visionvet.ai.ui.theme.*
import kotlinx.coroutines.launch

/**
 * Modern Settings Screen - Now FUNCTIONAL with DataStore!
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val settingsManager = remember { SettingsManager.getInstance(context) }
    
    // Collect settings from DataStore
    val darkModeEnabled by settingsManager.isDarkModeEnabled.collectAsState(initial = true)
    val notificationsEnabled by settingsManager.isNotificationsEnabled.collectAsState(initial = true)
    val autoSaveEnabled by settingsManager.isAutoSaveEnabled.collectAsState(initial = true)
    
    var showAboutDialog by remember { mutableStateOf(false) }
    var showClearDataDialog by remember { mutableStateOf(false) }
    var showSuccessSnackbar by remember { mutableStateOf(false) }
    
    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
        ) {
            // Top Bar
            SettingsTopBar(onNavigateBack = onNavigateBack)
            
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                // App Settings Section
                SettingsSection(title = "Uygulama") {
                    SettingsToggleItem(
                        icon = Icons.Outlined.DarkMode,
                        title = "Karanlık Mod",
                        subtitle = "Koyu tema kullan",
                        checked = darkModeEnabled,
                        onCheckedChange = { 
                            scope.launch { 
                                settingsManager.setDarkMode(it)
                                showSuccessSnackbar = true
                            }
                        }
                    )
                    
                    SettingsToggleItem(
                        icon = Icons.Outlined.Notifications,
                        title = "Bildirimler",
                        subtitle = "Analiz sonuçları için bildirim al",
                        checked = notificationsEnabled,
                        onCheckedChange = { 
                            scope.launch { 
                                settingsManager.setNotifications(it)
                                showSuccessSnackbar = true
                            }
                        }
                    )
                    
                    SettingsToggleItem(
                        icon = Icons.Outlined.Save,
                        title = "Otomatik Kaydet",
                        subtitle = "Sonuçları otomatik kaydet",
                        checked = autoSaveEnabled,
                        onCheckedChange = { 
                            scope.launch { 
                                settingsManager.setAutoSave(it)
                                showSuccessSnackbar = true
                            }
                        }
                    )
                }
        
                // Data Management Section
                SettingsSection(title = "Veri Yönetimi") {
                    SettingsClickableItem(
                        icon = Icons.Outlined.DeleteSweep,
                        title = "Tüm Ayarları Sıfırla",
                        subtitle = "Varsayılan ayarlara dön",
                        onClick = { showClearDataDialog = true },
                        iconColor = Color.Red
                    )
                }
                
                // Model Info Section
                SettingsSection(title = "Model Bilgisi") {
                    SettingsInfoItem(
                        icon = Icons.Outlined.Memory,
                        title = "Model Versiyonu",
                        value = "MobileNetV3-Large"
                    )
                    
                    SettingsInfoItem(
                        icon = Icons.Outlined.Category,
                        title = "Sınıf Sayısı",
                        value = "33 bakteri türü"
                    )
                    
                    SettingsInfoItem(
                        icon = Icons.Outlined.Speed,
                        title = "Çalışma Modu",
                        value = "On-Device (Offline)"
                    )
                }
        
                // About Section
                SettingsSection(title = "Hakkında") {
                    SettingsClickableItem(
                        icon = Icons.Outlined.Info,
                        title = "Uygulama Hakkında",
                        subtitle = "Versiyon 1.0.0",
                        onClick = { showAboutDialog = true }
                    )
                    
                    SettingsClickableItem(
                        icon = Icons.Outlined.Description,
                        title = "Gizlilik Politikası",
                        subtitle = "Veri kullanımı hakkında",
                        onClick = { /* Open privacy policy */ }
                    )
                    
                    SettingsClickableItem(
                        icon = Icons.Outlined.Help,
                        title = "Yardım & Destek",
                        subtitle = "SSS ve iletişim",
                        onClick = { /* Open help */ }
                    )
                }
                
                // Developer Credits
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 20.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "VisionVet AI",
                            style = MaterialTheme.typography.titleMedium,
                            color = BacteriaBlue
                        )
                        Text(
                            text = "Bakteriyel Koloni Analiz Sistemi",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "© 2024",
                            style = MaterialTheme.typography.labelSmall,
                            color = TextTertiary
                        )
                    }
                }
                
                // Bottom spacing
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
        
        // Success Snackbar
        if (showSuccessSnackbar) {
            LaunchedEffect(showSuccessSnackbar) {
                kotlinx.coroutines.delay(2000)
                showSuccessSnackbar = false
            }
            
            Snackbar(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(16.dp),
                containerColor = MicrobeGreen.copy(alpha = 0.9f),
                shape = RoundedCornerShape(12.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                    Text(
                        "Ayar kaydedildi",
                        color = Color.White,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
    
    
    // About Dialog
    if (showAboutDialog) {
        AlertDialog(
            onDismissRequest = { showAboutDialog = false },
            containerColor = DarkCard,
            title = {
                Text(
                    text = "VisionVet AI",
                    color = Color.White
                )
            },
            text = {
                Column {
                    Text(
                        text = "Bakteriyel koloni görüntülerini yapay zeka ile analiz eden mobil uygulama.",
                        color = TextSecondary
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "• 33 farklı bakteri türü tanıma\n• Offline çalışma desteği\n• Hızlı ve güvenilir sonuçlar",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = { showAboutDialog = false }) {
                    Text("Tamam", color = BacteriaBlue)
                }
            }
        )
    }
    
    // Clear Data Dialog
    if (showClearDataDialog) {
        AlertDialog(
            onDismissRequest = { showClearDataDialog = false },
            containerColor = DarkCard,
            shape = RoundedCornerShape(24.dp),
            title = {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(40.dp)
                            .background(
                                color = Color.Red.copy(alpha = 0.2f),
                                shape = CircleShape
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            Icons.Default.Warning,
                            contentDescription = null,
                            tint = Color.Red,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                    Text(
                        "Ayarları Sıfırla",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            },
            text = {
                Text(
                    "Tüm ayarlar varsayılan değerlere sıfırlanacak. Bu işlem geri alınamaz.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White.copy(alpha = 0.7f)
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        scope.launch {
                            settingsManager.clearAll()
                            showClearDataDialog = false
                            showSuccessSnackbar = true
                        }
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Red
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text("Sıfırla", fontWeight = FontWeight.SemiBold)
                }
            },
            dismissButton = {
                OutlinedButton(
                    onClick = { showClearDataDialog = false },
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = Color.White
                    )
                ) {
                    Text("İptal")
                }
            }
        )
    }
}

@Composable
private fun SettingsTopBar(onNavigateBack: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        DarkSurface,
                        DarkBackground
                    )
                )
            )
            .padding(top = 48.dp, bottom = 16.dp)
            .padding(horizontal = 16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            IconButton(
                onClick = onNavigateBack,
                modifier = Modifier
                    .size(44.dp)
                    .background(
                        color = Color.White.copy(alpha = 0.1f),
                        shape = CircleShape
                    )
            ) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White
                )
            }
            
            Text(
                "Ayarlar",
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontWeight = FontWeight.Bold
                ),
                color = Color.White
            )
        }
    }
}

@Composable
private fun SettingsSection(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleSmall,
            color = TextSecondary,
            modifier = Modifier.padding(start = 4.dp, bottom = 4.dp)
        )
        
        GlassmorphicCard(
            modifier = Modifier.fillMaxWidth(),
            cornerRadius = 20.dp
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(4.dp),
                content = content
            )
        }
    }
}

@Composable
private fun SettingsToggleItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.weight(1f)
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(
                        color = BacteriaBlue.copy(alpha = 0.1f),
                        shape = RoundedCornerShape(10.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = BacteriaBlue,
                    modifier = Modifier.size(22.dp)
                )
            }
            
            Column {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color.White
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
        }
        
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Color.White,
                checkedTrackColor = BacteriaBlue,
                uncheckedThumbColor = TextSecondary,
                uncheckedTrackColor = DarkCardBorder
            )
        )
    }
}

@Composable
private fun SettingsInfoItem(
    icon: ImageVector,
    title: String,
    value: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(
                        color = ElectricPurple.copy(alpha = 0.1f),
                        shape = RoundedCornerShape(10.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = ElectricPurple,
                    modifier = Modifier.size(22.dp)
                )
            }
            
            Text(
                text = title,
                style = MaterialTheme.typography.bodyLarge,
                color = Color.White
            )
        }
        
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = TextSecondary
        )
    }
}

@Composable
private fun SettingsClickableItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit,
    iconColor: Color = MicrobeGreen
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.weight(1f)
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(
                        color = iconColor.copy(alpha = 0.1f),
                        shape = RoundedCornerShape(10.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = iconColor,
                    modifier = Modifier.size(22.dp)
                )
            }
            
            Column {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color.White
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
        }
        
        IconButton(onClick = onClick) {
            Icon(
                imageVector = Icons.Filled.ChevronRight,
                contentDescription = null,
                tint = TextSecondary
            )
        }
    }
}
