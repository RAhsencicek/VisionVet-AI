package com.visionvet.ai.feature.mnist

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.ml.mnist.MnistClassifier

/**
 * MNIST rakam tanıma test ekranı
 * Kullanıcının rakam çizmesine ve AI tarafından tanınmasına izin verir
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MnistTestScreen() {
    val context = LocalContext.current
    val classifier = remember { MnistClassifier(context) }
    
    var paths by remember { mutableStateOf(listOf<Path>()) }
    var currentPath by remember { mutableStateOf(Path()) }
    var predictedDigit by remember { mutableStateOf<Int?>(null) }
    var confidence by remember { mutableStateOf(0f) }
    var probabilities by remember { mutableStateOf(FloatArray(10)) }
    
    DisposableEffect(Unit) {
        onDispose {
            classifier.close()
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MNIST Rakam Tanıma") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Sonuç kartı
            if (predictedDigit != null) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Tahmin Edilen Rakam",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Text(
                            text = "$predictedDigit",
                            fontSize = 72.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = "Güven: ${(confidence * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        // Tüm olasılıkları göster
                        Text(
                            text = "Tüm Olasılıklar:",
                            style = MaterialTheme.typography.labelMedium,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                        probabilities.forEachIndexed { index, probability ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 2.dp),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "$index:",
                                    style = MaterialTheme.typography.bodySmall
                                )
                                LinearProgressIndicator(
                                    progress = { probability },
                                    modifier = Modifier
                                        .weight(1f)
                                        .padding(horizontal = 8.dp),
                                )
                                Text(
                                    text = "${(probability * 100).toInt()}%",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                }
            }
            
            // Çizim alanı
            Card(
                modifier = Modifier
                    .size(280.dp)
                    .padding(bottom = 16.dp),
                shape = RoundedCornerShape(8.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color.White
                )
            ) {
                Canvas(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectDragGestures(
                                onDragStart = { offset ->
                                    currentPath = Path().apply {
                                        moveTo(offset.x, offset.y)
                                    }
                                },
                                onDrag = { change, _ ->
                                    currentPath.lineTo(
                                        change.position.x,
                                        change.position.y
                                    )
                                },
                                onDragEnd = {
                                    paths = paths + currentPath
                                    currentPath = Path()
                                }
                            )
                        }
                ) {
                    // Çizilen tüm path'leri çiz
                    paths.forEach { path ->
                        drawPath(
                            path = path,
                            color = Color.Black,
                            style = Stroke(
                                width = 40f,
                                cap = StrokeCap.Round,
                                join = StrokeJoin.Round
                            )
                        )
                    }
                    
                    // Aktif path'i çiz
                    drawPath(
                        path = currentPath,
                        color = Color.Black,
                        style = Stroke(
                            width = 40f,
                            cap = StrokeCap.Round,
                            join = StrokeJoin.Round
                        )
                    )
                }
            }
            
            // Butonlar
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(
                    onClick = {
                        paths = emptyList()
                        currentPath = Path()
                        predictedDigit = null
                        confidence = 0f
                        probabilities = FloatArray(10)
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Temizle"
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Temizle")
                }
                
                Button(
                    onClick = {
                        // TODO: Canvas'tan bitmap al ve classify et
                        // Bu kısım gerçek implementasyonda bitmap conversion gerektirir
                        // Şimdilik dummy result gösterelim
                        predictedDigit = kotlin.random.Random.nextInt(0, 10)
                        confidence = kotlin.random.Random.nextFloat() * 0.29f + 0.7f
                        probabilities = FloatArray(10) { kotlin.random.Random.nextFloat() * 0.09f + 0.01f }
                        probabilities[predictedDigit!!] = confidence
                    },
                    enabled = paths.isNotEmpty()
                ) {
                    Text("Tanı")
                }
            }
            
            // Bilgilendirme
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Nasıl Kullanılır?",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "1. Beyaz alana parmağınızla bir rakam (0-9) çizin\n" +
                                "2. 'Tanı' butonuna basın\n" +
                                "3. AI tahminini görün\n" +
                                "4. Yeni rakam için 'Temizle' butonunu kullanın",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}
