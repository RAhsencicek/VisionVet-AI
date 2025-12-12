package com.visionvet.ai.navigation

sealed class Screen(val route: String) {
    object Login : Screen("login")
    object Home : Screen("home") // Ana TabView container
    object Dashboard : Screen("dashboard")
    object Camera : Screen("camera")
    object AnalysisResult : Screen("analysis_result/{analysisId}") {
        fun createRoute(analysisId: String) = "analysis_result/$analysisId"
    }
    object AnalysisHistory : Screen("analysis_history")
    object Profile : Screen("profile")
    object Settings : Screen("settings")
    object MnistTest : Screen("mnist_test") // MNIST rakam tanıma test ekranı
    object BacterialScan : Screen("bacterial_scan") // Bakteriyel koloni tarama ekranı
    object BacterialTest : Screen("bacterial_test") // Bakteriyel test ekranı
    object BacterialResult : Screen("bacterial_result/{resultId}") {
        fun createRoute(resultId: String) = "bacterial_result/$resultId"
    }
}
