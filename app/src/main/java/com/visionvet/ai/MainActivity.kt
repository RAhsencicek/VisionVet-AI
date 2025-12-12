package com.visionvet.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.visionvet.ai.navigation.Screen
import com.visionvet.ai.feature.dashboard.DashboardScreen
import com.visionvet.ai.feature.mnist.MnistTestScreen
import com.visionvet.ai.feature.bacterial.BacterialTestScreen
import com.visionvet.ai.feature.bacterial.BacterialScanScreen
import com.visionvet.ai.feature.bacterial.BacterialResultScreen
import com.visionvet.ai.feature.bacterial.BacterialHistoryScreen
import com.visionvet.ai.feature.home.HomeView
import com.visionvet.ai.feature.settings.SettingsScreen
import com.visionvet.ai.ui.screens.login.LoginScreen
import com.visionvet.ai.ui.theme.VisionVetAITheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VisionVetAITheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    OpCaApp()
                }
            }
        }
    }
}

@Composable
fun OpCaApp() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = Screen.Home.route
    ) {
        composable(Screen.Login.route) {
            LoginScreen(
                onLoginSuccess = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Login.route) { inclusive = true }
                    }
                }
            )
        }

        composable(Screen.Home.route) {
            HomeView(
                onLogout = {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(Screen.Home.route) { inclusive = true }
                    }
                }
            )
        }

        composable(Screen.Dashboard.route) {
            DashboardScreen(
                onNavigateToMnist = {
                    navController.navigate(Screen.MnistTest.route)
                },
                onNavigateToBacterialTest = {
                    navController.navigate(Screen.BacterialTest.route)
                },
                onNavigateToBacterialScan = {
                    navController.navigate(Screen.BacterialScan.route)
                }
            )
        }
        
        composable(Screen.Settings.route) {
            SettingsScreen(
                onLogout = {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(Screen.Home.route) { inclusive = true }
                    }
                }
            )
        }
        
        composable(Screen.MnistTest.route) {
            MnistTestScreen()
        }
        
        composable(Screen.BacterialScan.route) {
            BacterialScanScreen(
                onNavigateToResult = { resultId ->
                    navController.navigate(Screen.BacterialResult.createRoute(resultId))
                },
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
        
        composable(Screen.BacterialResult.route) { backStackEntry ->
            val resultId = backStackEntry.arguments?.getString("resultId") ?: ""
            BacterialResultScreen(
                resultId = resultId,
                onNavigateBack = {
                    navController.popBackStack()
                },
                onNavigateToHistory = {
                    // TODO: Navigate to bacterial history
                    navController.popBackStack()
                }
            )
        }
        
        composable(Screen.BacterialTest.route) {
            BacterialTestScreen()
        }
    }
}
