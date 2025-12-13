package com.visionvet.ai.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat

/**
 * VisionVet-AI Dark Color Scheme
 * Modern, vibrant dark theme with cyan accents
 */
private val DarkColorScheme = darkColorScheme(
    // Primary
    primary = BacteriaBlue,
    onPrimary = Color.Black,
    primaryContainer = DeepBlue,
    onPrimaryContainer = Color.White,
    
    // Secondary
    secondary = ElectricPurple,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFF4A148C),
    onSecondaryContainer = Color.White,
    
    // Tertiary
    tertiary = MicrobeGreen,
    onTertiary = Color.Black,
    tertiaryContainer = Color(0xFF004D40),
    onTertiaryContainer = Color.White,
    
    // Background & Surface
    background = DarkBackground,
    onBackground = TextPrimary,
    surface = DarkSurface,
    onSurface = TextPrimary,
    surfaceVariant = DarkCard,
    onSurfaceVariant = TextSecondary,
    
    // Others
    error = AlertRed,
    onError = Color.White,
    errorContainer = Color(0xFF93000A),
    onErrorContainer = Color.White,
    outline = DarkCardBorder,
    outlineVariant = Color(0xFF2D3748),
    inverseSurface = LightSurface,
    inverseOnSurface = TextDark,
    inversePrimary = DeepBlue,
    surfaceTint = BacteriaBlue
)

/**
 * VisionVet-AI Light Color Scheme
 */
private val LightColorScheme = lightColorScheme(
    // Primary
    primary = DeepBlue,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD1E4FF),
    onPrimaryContainer = Color(0xFF001D36),
    
    // Secondary
    secondary = ElectricPurple,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFE8DEF8),
    onSecondaryContainer = Color(0xFF1D192B),
    
    // Tertiary
    tertiary = MicrobeGreen,
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFD0F8D6),
    onTertiaryContainer = Color(0xFF002106),
    
    // Background & Surface
    background = LightBackground,
    onBackground = TextLight,
    surface = LightSurface,
    onSurface = TextLight,
    surfaceVariant = LightCard,
    onSurfaceVariant = TextTertiary,
    
    // Others
    error = AlertRed,
    onError = Color.White,
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF410002),
    outline = LightCardBorder,
    outlineVariant = Color(0xFFCAC4D0)
)

/**
 * Custom Shapes for VisionVet-AI
 */
val VisionVetShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(24.dp),
    extraLarge = RoundedCornerShape(32.dp)
)

/**
 * VisionVet-AI Theme
 * @param darkTheme Whether to use dark theme (default: follow system)
 * @param dynamicColor Whether to use dynamic colors on Android 12+ (disabled for consistent branding)
 */
@Composable
fun VisionVetAITheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false, // Disabled for consistent branding
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    
    // Set status bar color
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        shapes = VisionVetShapes,
        content = content
    )
}

/**
 * Extension properties for common color access
 */
object VisionVetColors {
    val cyan @Composable get() = BacteriaBlue
    val purple @Composable get() = ElectricPurple
    val green @Composable get() = MicrobeGreen
    val orange @Composable get() = LabOrange
    val red @Composable get() = AlertRed
    
    val cardBackground @Composable get() = if (isSystemInDarkTheme()) DarkCard else LightCard
    val cardBorder @Composable get() = if (isSystemInDarkTheme()) DarkCardBorder else LightCardBorder
}