package com.visionvet.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.visionvet.ai.navigation.BottomNavItem
import com.visionvet.ai.navigation.Screen
import com.visionvet.ai.feature.dashboard.DashboardScreen
import com.visionvet.ai.feature.bacterial.BacterialScanScreen
import com.visionvet.ai.feature.bacterial.BacterialResultScreen
import com.visionvet.ai.feature.bacterial.BacterialHistoryScreen
import com.visionvet.ai.feature.settings.SettingsScreen
import com.visionvet.ai.feature.splash.SplashScreen
import com.visionvet.ai.ui.components.ModernBottomNavigation
import com.visionvet.ai.ui.theme.DarkBackground
import com.visionvet.ai.ui.theme.VisionVetAITheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VisionVetAITheme(darkTheme = true) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    VisionVetApp()
                }
            }
        }
    }
}

@Composable
fun VisionVetApp() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    
    // Screens that should show bottom navigation
    val bottomNavScreens = listOf(
        Screen.Dashboard.route,
        Screen.Scan.route,
        Screen.History.route,
        Screen.Settings.route
    )
    
    val showBottomNav = currentRoute in bottomNavScreens

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        bottomBar = {
            AnimatedVisibility(
                visible = showBottomNav,
                enter = slideInVertically(
                    initialOffsetY = { it },
                    animationSpec = tween(300)
                ) + fadeIn(animationSpec = tween(300)),
                exit = slideOutVertically(
                    targetOffsetY = { it },
                    animationSpec = tween(300)
                ) + fadeOut(animationSpec = tween(300))
            ) {
                ModernBottomNavigation(
                    items = BottomNavItem.items,
                    selectedRoute = currentRoute ?: Screen.Dashboard.route,
                    onItemSelected = { item ->
                        navController.navigate(item.route) {
                            popUpTo(navController.graph.findStartDestination().id) {
                                saveState = true
                            }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                )
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = Screen.Splash.route,
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(MaterialTheme.colorScheme.background)
        ) {
            // Splash Screen
            composable(
                route = Screen.Splash.route,
                enterTransition = { fadeIn(animationSpec = tween(300)) },
                exitTransition = { fadeOut(animationSpec = tween(300)) }
            ) {
                SplashScreen(
                    onSplashFinished = {
                        navController.navigate(Screen.Dashboard.route) {
                            popUpTo(Screen.Splash.route) { inclusive = true }
                        }
                    }
                )
            }
            
            // Dashboard (Home)
            composable(
                route = Screen.Dashboard.route,
                enterTransition = { 
                    fadeIn(animationSpec = tween(300)) + 
                    slideInHorizontally(animationSpec = tween(300)) 
                },
                exitTransition = { fadeOut(animationSpec = tween(200)) }
            ) {
                DashboardScreen(
                    onNavigateToScan = {
                        navController.navigate(Screen.Scan.route)
                    },
                    onNavigateToHistory = {
                        navController.navigate(Screen.History.route)
                    }
                )
            }
            
            // Scan Screen
            composable(
                route = Screen.Scan.route,
                enterTransition = { 
                    fadeIn(animationSpec = tween(300)) + 
                    scaleIn(initialScale = 0.95f, animationSpec = tween(300))
                },
                exitTransition = { fadeOut(animationSpec = tween(200)) }
            ) {
                BacterialScanScreen(
                    onNavigateToResult = { resultId ->
                        navController.navigate(Screen.BacterialResult.createRoute(resultId))
                    },
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            
            // History Screen
            composable(
                route = Screen.History.route,
                enterTransition = { 
                    fadeIn(animationSpec = tween(300)) + 
                    slideInHorizontally(animationSpec = tween(300)) 
                },
                exitTransition = { fadeOut(animationSpec = tween(200)) }
            ) {
                BacterialHistoryScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onResultClick = { resultId ->
                        navController.navigate(Screen.BacterialResult.createRoute(resultId))
                    }
                )
            }
            
            // Settings Screen
            composable(
                route = Screen.Settings.route,
                enterTransition = { 
                    fadeIn(animationSpec = tween(300)) + 
                    slideInHorizontally(animationSpec = tween(300)) 
                },
                exitTransition = { fadeOut(animationSpec = tween(200)) }
            ) {
                SettingsScreen(
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            
            // Bacterial Result Screen (Detail)
            composable(
                route = Screen.BacterialResult.route,
                enterTransition = { 
                    fadeIn(animationSpec = tween(300)) + 
                    slideInVertically(
                        initialOffsetY = { it / 2 },
                        animationSpec = tween(300)
                    )
                },
                exitTransition = { 
                    fadeOut(animationSpec = tween(200)) +
                    slideOutVertically(
                        targetOffsetY = { it / 2 },
                        animationSpec = tween(200)
                    )
                }
            ) { backStackEntry ->
                val resultId = backStackEntry.arguments?.getString("resultId") ?: ""
                BacterialResultScreen(
                    resultId = resultId,
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onNavigateToHistory = {
                        navController.navigate(Screen.History.route) {
                            popUpTo(Screen.Dashboard.route)
                        }
                    }
                )
            }
        }
    }
}
