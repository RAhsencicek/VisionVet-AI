package com.visionvet.ai.feature.splash

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.ui.theme.*
import kotlinx.coroutines.delay
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Modern Animated Splash Screen
 * Features: Animated bacteria particles, DNA helix effect, gradient background
 */
@Composable
fun SplashScreen(
    onSplashFinished: () -> Unit
) {
    var startAnimation by remember { mutableStateOf(false) }
    
    // Main animation timeline
    LaunchedEffect(key1 = true) {
        startAnimation = true
        delay(2500) // Show splash for 2.5 seconds
        onSplashFinished()
    }
    
    // Alpha animation for fade in
    val alphaAnim by animateFloatAsState(
        targetValue = if (startAnimation) 1f else 0f,
        animationSpec = tween(durationMillis = 800, easing = EaseOutQuart),
        label = "alpha"
    )
    
    // Scale animation for logo
    val scaleAnim by animateFloatAsState(
        targetValue = if (startAnimation) 1f else 0.5f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "scale"
    )
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        DarkBackground,
                        Color(0xFF0A1628),
                        DarkSurface
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        // Animated background particles
        AnimatedBackgroundParticles()
        
        // DNA Helix Animation
        DNAHelixAnimation(
            modifier = Modifier
                .fillMaxSize()
                .alpha(0.3f)
        )
        
        // Main content
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .alpha(alphaAnim)
                .scale(scaleAnim)
        ) {
            // Animated bacteria icon
            AnimatedBacteriaIcon(
                modifier = Modifier.size(150.dp)
            )
            
            Spacer(modifier = Modifier.height(32.dp))
            
            // App name with gradient
            Text(
                text = "VisionVet",
                style = MaterialTheme.typography.displaySmall.copy(
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                ),
                color = Color.White
            )
            
            // Subtitle with cyan glow effect
            Text(
                text = "AI",
                style = MaterialTheme.typography.headlineLarge.copy(
                    fontWeight = FontWeight.Light,
                    letterSpacing = 8.sp
                ),
                color = BacteriaBlue
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Tagline
            Text(
                text = "Bakteriyel Koloni Analizi",
                style = MaterialTheme.typography.bodyLarge,
                color = TextSecondary
            )
            
            Spacer(modifier = Modifier.height(48.dp))
            
            // Loading indicator
            PulsingLoadingIndicator()
        }
    }
}

/**
 * Animated bacteria icon with rotation and pulse
 */
@Composable
fun AnimatedBacteriaIcon(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "bacteria")
    
    // Rotation animation
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(8000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )
    
    // Pulse animation
    val pulse by infiniteTransition.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )
    
    // Glow animation
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.7f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
    )
    
    Canvas(modifier = modifier.scale(pulse)) {
        val center = Offset(size.width / 2, size.height / 2)
        val radius = size.minDimension / 3
        
        // Outer glow
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(
                    BacteriaBlue.copy(alpha = glowAlpha),
                    Color.Transparent
                ),
                center = center,
                radius = radius * 2
            ),
            radius = radius * 2,
            center = center
        )
        
        // Main bacteria body
        rotate(rotation, pivot = center) {
            // Central body
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        BacteriaBlue,
                        DeepBlue
                    ),
                    center = center.copy(x = center.x - radius * 0.2f)
                ),
                radius = radius,
                center = center
            )
            
            // Flagella (bacteria tails)
            for (i in 0 until 6) {
                val angle = (i * 60f + rotation * 0.5f) * (Math.PI / 180f)
                val startX = center.x + radius * cos(angle).toFloat()
                val startY = center.y + radius * sin(angle).toFloat()
                
                val path = Path().apply {
                    moveTo(startX, startY)
                    val control1X = startX + radius * 0.8f * cos(angle + 0.3).toFloat()
                    val control1Y = startY + radius * 0.8f * sin(angle + 0.3).toFloat()
                    val endX = startX + radius * 1.2f * cos(angle).toFloat()
                    val endY = startY + radius * 1.2f * sin(angle).toFloat()
                    quadraticBezierTo(control1X, control1Y, endX, endY)
                }
                
                drawPath(
                    path = path,
                    color = MicrobeGreen.copy(alpha = 0.8f),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(
                        width = 4f,
                        cap = androidx.compose.ui.graphics.StrokeCap.Round
                    )
                )
            }
            
            // Inner detail circles
            drawCircle(
                color = Color.White.copy(alpha = 0.2f),
                radius = radius * 0.3f,
                center = center.copy(
                    x = center.x - radius * 0.3f,
                    y = center.y - radius * 0.2f
                )
            )
        }
    }
}

/**
 * DNA Helix animation in background
 */
@Composable
fun DNAHelixAnimation(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "dna")
    
    val offset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 100f,
        animationSpec = infiniteRepeatable(
            animation = tween(3000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "dnaOffset"
    )
    
    Canvas(modifier = modifier) {
        val width = size.width
        val height = size.height
        val amplitude = width * 0.15f
        val centerX = width / 2
        
        // Draw double helix
        for (y in 0..height.toInt() step 20) {
            val adjustedY = y + offset
            val x1 = centerX + amplitude * sin((adjustedY * 0.02f).toDouble()).toFloat()
            val x2 = centerX - amplitude * sin((adjustedY * 0.02f).toDouble()).toFloat()
            
            // First strand
            drawCircle(
                color = BacteriaBlue.copy(alpha = 0.4f),
                radius = 4f,
                center = Offset(x1, y.toFloat())
            )
            
            // Second strand
            drawCircle(
                color = MicrobeGreen.copy(alpha = 0.4f),
                radius = 4f,
                center = Offset(x2, y.toFloat())
            )
            
            // Connecting lines (base pairs)
            if (y % 40 == 0) {
                drawLine(
                    color = Color.White.copy(alpha = 0.1f),
                    start = Offset(x1, y.toFloat()),
                    end = Offset(x2, y.toFloat()),
                    strokeWidth = 2f
                )
            }
        }
    }
}

/**
 * Animated floating particles in background
 */
@Composable
fun AnimatedBackgroundParticles() {
    val particles = remember {
        List(30) {
            Particle(
                x = Random.nextFloat(),
                y = Random.nextFloat(),
                radius = Random.nextFloat() * 4f + 2f,
                speed = Random.nextFloat() * 0.5f + 0.1f,
                alpha = Random.nextFloat() * 0.5f + 0.1f
            )
        }
    }
    
    val infiniteTransition = rememberInfiniteTransition(label = "particles")
    
    val animatedOffset by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(10000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "particleOffset"
    )
    
    Canvas(modifier = Modifier.fillMaxSize()) {
        particles.forEach { particle ->
            val yOffset = (particle.y + animatedOffset * particle.speed) % 1f
            drawCircle(
                color = BacteriaBlue.copy(alpha = particle.alpha),
                radius = particle.radius,
                center = Offset(
                    x = particle.x * size.width,
                    y = yOffset * size.height
                )
            )
        }
    }
}

/**
 * Pulsing loading indicator
 */
@Composable
fun PulsingLoadingIndicator() {
    val infiniteTransition = rememberInfiniteTransition(label = "loading")
    
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(3) { index ->
            val delay = index * 200
            
            val scale by infiniteTransition.animateFloat(
                initialValue = 0.5f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(
                        durationMillis = 600,
                        delayMillis = delay,
                        easing = EaseInOutSine
                    ),
                    repeatMode = RepeatMode.Reverse
                ),
                label = "dot$index"
            )
            
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .scale(scale)
                    .background(
                        color = BacteriaBlue,
                        shape = MaterialTheme.shapes.extraLarge
                    )
            )
        }
    }
}

private data class Particle(
    val x: Float,
    val y: Float,
    val radius: Float,
    val speed: Float,
    val alpha: Float
)

// Easing functions
private val EaseOutQuart: Easing = Easing { fraction ->
    1 - (1 - fraction).let { it * it * it * it }
}

private val EaseInOutSine: Easing = Easing { fraction ->
    -(cos(Math.PI * fraction).toFloat() - 1) / 2
}
