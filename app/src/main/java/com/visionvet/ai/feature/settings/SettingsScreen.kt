package com.visionvet.ai.feature.settings

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.ui.theme.VisionVetAITheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onLogout: () -> Unit
) {
    var showLogoutDialog by remember { mutableStateOf(false) }
    var notificationsEnabled by remember { mutableStateOf(true) }
    var autoUploadEnabled by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Header
        Text(
            text = "Settings",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 24.dp)
        )

        // Profile Section
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                Text(
                    text = "Profile",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                SettingsItem(
                    icon = Icons.Default.Person,
                    title = "Account Information",
                    subtitle = "Manage your profile details",
                    onClick = { /* Navigate to profile */ }
                )

                SettingsItem(
                    icon = Icons.Default.Lock, // Security yerine Lock
                    title = "Privacy & Security",
                    subtitle = "Password, data protection",
                    onClick = { /* Navigate to privacy */ }
                )
            }
        }

        // Preferences Section
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                Text(
                    text = "Preferences",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                SettingsToggleItem(
                    icon = Icons.Default.Notifications,
                    title = "Notifications",
                    subtitle = "Get notified about analysis results",
                    isChecked = notificationsEnabled,
                    onToggle = { notificationsEnabled = it }
                )

                SettingsToggleItem(
                    icon = Icons.Default.CloudUpload,
                    title = "Auto Upload",
                    subtitle = "Automatically upload analyses to cloud",
                    isChecked = autoUploadEnabled,
                    onToggle = { autoUploadEnabled = it }
                )

                SettingsItem(
                    icon = Icons.Default.Palette,
                    title = "Appearance",
                    subtitle = "Theme, colors, display options",
                    onClick = { /* Navigate to appearance */ }
                )
            }
        }

        // Data & Storage Section
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                Text(
                    text = "Data & Storage",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                SettingsItem(
                    icon = Icons.Default.Storage,
                    title = "Data Management",
                    subtitle = "Export, backup, clear data",
                    onClick = { /* Navigate to data management */ }
                )

                SettingsItem(
                    icon = Icons.Default.CloudSync,
                    title = "Sync Settings",
                    subtitle = "Configure cloud synchronization",
                    onClick = { /* Navigate to sync settings */ }
                )
            }
        }

        // Support Section
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 20.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                Text(
                    text = "Support",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                SettingsItem(
                    icon = Icons.Default.Help,
                    title = "Help & Support",
                    subtitle = "FAQs, contact support",
                    onClick = { /* Navigate to help */ }
                )

                SettingsItem(
                    icon = Icons.Default.Info,
                    title = "About",
                    subtitle = "Version, licenses, credits",
                    onClick = { /* Navigate to about */ }
                )
            }
        }

        // Logout Section
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.1f)
            )
        ) {
            SettingsItem(
                icon = Icons.AutoMirrored.Filled.ExitToApp,
                title = "Logout",
                subtitle = "Sign out of your account",
                onClick = { showLogoutDialog = true },
                titleColor = MaterialTheme.colorScheme.error
            )
        }
    }

    // Logout confirmation dialog
    if (showLogoutDialog) {
        AlertDialog(
            onDismissRequest = { showLogoutDialog = false },
            title = { Text("Logout") },
            text = { Text("Are you sure you want to logout?") },
            confirmButton = {
                TextButton(
                    onClick = {
                        showLogoutDialog = false
                        onLogout()
                    }
                ) {
                    Text("Logout", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showLogoutDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun SettingsItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit,
    titleColor: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = titleColor,
            modifier = Modifier
                .size(24.dp)
                .padding(end = 16.dp)
        )

        Column(
            modifier = Modifier.weight(1f)
        ) {
            Text(
                text = title,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = titleColor
            )
            Text(
                text = subtitle,
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Icon(
            Icons.AutoMirrored.Filled.ArrowForward,
            contentDescription = "Navigate",
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(20.dp)
        )
    }
}

@Composable
private fun SettingsToggleItem(
    icon: ImageVector,
    title: String,
    subtitle: String,
    isChecked: Boolean,
    onToggle: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier
                .size(24.dp)
                .padding(end = 16.dp)
        )

        Column(
            modifier = Modifier.weight(1f)
        ) {
            Text(
                text = title,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = subtitle,
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Switch(
            checked = isChecked,
            onCheckedChange = onToggle
        )
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun SettingsScreenPreview() {
    VisionVetAITheme {
        SettingsScreen(
            onLogout = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun SettingsScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        SettingsScreen(
            onLogout = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun SettingsItemPreview() {
    VisionVetAITheme {
        Column(modifier = Modifier.padding(16.dp)) {
            SettingsItem(
                icon = Icons.Default.Person,
                title = "Account Information",
                subtitle = "Manage your profile details",
                onClick = {}
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun SettingsToggleItemPreview() {
    VisionVetAITheme {
        Column(modifier = Modifier.padding(16.dp)) {
            SettingsToggleItem(
                icon = Icons.Default.Notifications,
                title = "Notifications",
                subtitle = "Get notified about analysis results",
                isChecked = true,
                onToggle = {}
            )
        }
    }
}
