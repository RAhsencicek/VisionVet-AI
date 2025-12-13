package com.visionvet.ai.ui.theme

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

/**
 * VisionVet-AI Modern Color Palette
 * Medical/Scientific theme with vibrant gradients
 */

// ============== Primary Colors ==============
val BacteriaBlue = Color(0xFF00D4FF)         // Cyan - Primary
val BacteriaBlueDark = Color(0xFF0099CC)     // Darker cyan
val DeepBlue = Color(0xFF0066FF)             // Deep blue accent
val ElectricPurple = Color(0xFF7C3AED)       // Electric purple

// ============== Accent Colors ==============
val MicrobeGreen = Color(0xFF00D68F)         // Success/Positive
val NeonGreen = Color(0xFF39FF14)            // Highlight
val LabOrange = Color(0xFFFF8C00)            // Warning
val AlertRed = Color(0xFFFF3B30)             // Error/Danger
val GoldenYellow = Color(0xFFFFD700)         // Premium accent

// ============== Dark Theme Colors ==============
val DarkBackground = Color(0xFF0A0E17)       // Very dark blue-black
val DarkSurface = Color(0xFF111827)          // Card background
val DarkCard = Color(0xFF1F2937)             // Elevated card
val DarkCardBorder = Color(0xFF374151)       // Card border
val DarkSurfaceVariant = Color(0xFF1E2533)   // Alternative surface

// ============== Light Theme Colors ==============
val LightBackground = Color(0xFFF8FAFC)      // Clean white-blue
val LightSurface = Color(0xFFFFFFFF)         // Pure white
val LightCard = Color(0xFFF1F5F9)            // Light card
val LightCardBorder = Color(0xFFE2E8F0)      // Light border

// ============== Text Colors ==============
val TextPrimary = Color(0xFFFFFFFF)          // White text
val TextSecondary = Color(0xFF9CA3AF)        // Gray text
val TextTertiary = Color(0xFF6B7280)         // Muted text
val TextDark = Color(0xFF111827)             // Dark mode text
val TextLight = Color(0xFF1F2937)            // Light mode primary text

// ============== Gradient Definitions ==============
val CyanGradient = Brush.horizontalGradient(
    colors = listOf(BacteriaBlue, DeepBlue)
)

val PurpleGradient = Brush.horizontalGradient(
    colors = listOf(Color(0xFF667EEA), Color(0xFF764BA2))
)

val SunsetGradient = Brush.horizontalGradient(
    colors = listOf(Color(0xFFFF6B6B), Color(0xFFFFE66D))
)

val NeonGradient = Brush.horizontalGradient(
    colors = listOf(BacteriaBlue, MicrobeGreen)
)

val DarkGradient = Brush.verticalGradient(
    colors = listOf(DarkBackground, DarkSurface)
)

val GlassmorphicGradient = Brush.linearGradient(
    colors = listOf(
        Color.White.copy(alpha = 0.15f),
        Color.White.copy(alpha = 0.05f)
    )
)

// ============== Glassmorphism Colors ==============
val GlassWhite = Color.White.copy(alpha = 0.1f)
val GlassBorder = Color.White.copy(alpha = 0.2f)
val GlassHighlight = Color.White.copy(alpha = 0.3f)

// ============== Status Colors ==============
val StatusSuccess = MicrobeGreen
val StatusWarning = LabOrange
val StatusError = AlertRed
val StatusInfo = BacteriaBlue

// ============== Legacy Colors (for compatibility) ==============
val Purple80 = Color(0xFFD0BCFF)
val PurpleGrey80 = Color(0xFFCCC2DC)
val Pink80 = Color(0xFFEFB8C8)
val Purple40 = Color(0xFF6650a4)
val PurpleGrey40 = Color(0xFF625b71)
val Pink40 = Color(0xFF7D5260)