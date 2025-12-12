package com.visionvet.ai.feature.home

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.visionvet.ai.feature.dashboard.DashboardScreen
import com.visionvet.ai.feature.scanner.NewScanScreen
import com.visionvet.ai.feature.history.HistoryScreen
import com.visionvet.ai.feature.settings.SettingsScreen
import com.visionvet.ai.feature.mnist.MnistTestScreen
import com.visionvet.ai.feature.bacterial.BacterialTestScreen
import com.visionvet.ai.feature.bacterial.BacterialScanScreen
import com.visionvet.ai.feature.bacterial.BacterialResultScreen
import com.visionvet.ai.feature.bacterial.BacterialHistoryScreen
import com.visionvet.ai.ui.theme.VisionVetAITheme

// Bottom Navigation Items
sealed class BottomNavItem(
    val route: String,
    val title: String,
    val icon: ImageVector
) {
    object Dashboard : BottomNavItem("dashboard", "Dashboard", Icons.Default.Home)
    object NewScan : BottomNavItem("new_scan", "New Scan", Icons.Default.Add)
    object History : BottomNavItem("history", "History", Icons.Default.DateRange)
    object Settings : BottomNavItem("settings", "Settings", Icons.Default.Settings)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeView(
    onLogout: () -> Unit = {}
) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf(
                    BottomNavItem.Dashboard,
                    BottomNavItem.NewScan,
                    BottomNavItem.History,
                    BottomNavItem.Settings
                ).forEach { item ->
                    NavigationBarItem(
                        icon = { Icon(item.icon, contentDescription = item.title) },
                        label = { Text(item.title) },
                        selected = currentRoute == item.route,
                        onClick = {
                            navController.navigate(item.route) {
                                popUpTo(navController.graph.startDestinationId)
                                launchSingleTop = true
                            }
                        }
                    )
                }
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = BottomNavItem.Dashboard.route,
            modifier = Modifier.padding(paddingValues)
        ) {
            composable(BottomNavItem.Dashboard.route) {
                DashboardScreen(
                    onNavigateToCamera = { /* Navigate to camera */ },
                    onNavigateToHistory = { navController.navigate(BottomNavItem.History.route) },
                    onNavigateToProfile = { navController.navigate(BottomNavItem.Settings.route) },
                    onAnalysisClick = { /* Navigate to analysis detail */ },
                    onNavigateToMnist = { navController.navigate("mnist_test") },
                    onNavigateToBacterialTest = { navController.navigate("bacterial_test") },
                    onNavigateToBacterialScan = { navController.navigate("bacterial_scan") },
                    onNavigateToBacterialHistory = { navController.navigate("bacterial_history") }
                )
            }

            composable(BottomNavItem.NewScan.route) {
                NewScanScreen(
                    onNavigateToCamera = { /* Navigate to camera with type */ },
                    onNavigateToDrawing = { /* Navigate to drawing */ }
                )
            }

            composable(BottomNavItem.History.route) {
                HistoryScreen(
                    analyses = emptyList(), // TODO: Implement ViewModel
                    onAnalysisClick = { /* Navigate to analysis detail */ }
                )
            }

            composable(BottomNavItem.Settings.route) {
                SettingsScreen(
                    onLogout = onLogout
                )
            }
            
            // ML Test Screens
            composable("mnist_test") {
                MnistTestScreen()
            }
            
            composable("bacterial_test") {
                BacterialTestScreen()
            }
            
            composable("bacterial_scan") {
                BacterialScanScreen(
                    onNavigateToResult = { resultId ->
                        navController.navigate("bacterial_result/$resultId")
                    },
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            
            composable("bacterial_result/{resultId}") { backStackEntry ->
                val resultId = backStackEntry.arguments?.getString("resultId") ?: ""
                BacterialResultScreen(
                    resultId = resultId,
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onNavigateToHistory = {
                        navController.navigate("bacterial_history")
                    }
                )
            }
            
            composable("bacterial_history") {
                BacterialHistoryScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onResultClick = { resultId ->
                        navController.navigate("bacterial_result/$resultId")
                    }
                )
            }
        }
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun HomeViewPreview() {
    VisionVetAITheme {
        HomeView(
            onLogout = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun HomeViewDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        HomeView(
            onLogout = {}
        )
    }
}
