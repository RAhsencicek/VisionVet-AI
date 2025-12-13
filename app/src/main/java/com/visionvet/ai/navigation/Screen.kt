package com.visionvet.ai.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * VisionVet-AI Navigation Routes
 * Simplified structure for bacterial analysis only
 */
sealed class Screen(val route: String) {
    // Splash Screen - App entry point with animation
    object Splash : Screen("splash")
    
    // Main Screens
    object Dashboard : Screen("dashboard")
    object Scan : Screen("scan")
    object History : Screen("history")
    object Settings : Screen("settings")
    
    // Detail Screens
    object BacterialResult : Screen("bacterial_result/{resultId}") {
        fun createRoute(resultId: String) = "bacterial_result/$resultId"
    }
}

/**
 * Bottom Navigation Items
 */
sealed class BottomNavItem(
    val route: String,
    val title: String,
    val icon: ImageVector,
    val selectedIcon: ImageVector
) {
    object Dashboard : BottomNavItem(
        route = Screen.Dashboard.route,
        title = "Ana Sayfa",
        icon = Icons.Outlined.Home,
        selectedIcon = Icons.Filled.Home
    )
    
    object Scan : BottomNavItem(
        route = Screen.Scan.route,
        title = "Tara",
        icon = Icons.Outlined.CameraAlt,
        selectedIcon = Icons.Filled.CameraAlt
    )
    
    object History : BottomNavItem(
        route = Screen.History.route,
        title = "Geçmiş",
        icon = Icons.Outlined.History,
        selectedIcon = Icons.Filled.History
    )
    
    object Settings : BottomNavItem(
        route = Screen.Settings.route,
        title = "Ayarlar",
        icon = Icons.Outlined.Settings,
        selectedIcon = Icons.Filled.Settings
    )
    
    companion object {
        val items = listOf(Dashboard, Scan, History, Settings)
    }
}
