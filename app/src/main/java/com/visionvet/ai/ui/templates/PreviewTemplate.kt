package com.visionvet.ai.ui.templates

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.ui.theme.VisionVetAITheme

/**
 * PREVIEW TEMPLATE - Swift benzeri preview sistemi
 *
 * Bu template'i yeni ekranlar oluştururken kullanın.
 *
 * Swift'teki #Preview gibi çalışan Jetpack Compose preview'ları için:
 * 1. @Preview annotation'ını import edin
 * 2. Her ana component için preview fonksiyonu oluşturun
 * 3. Farklı durumlar için preview varyantları ekleyin
 * 4. VisionVetAITheme wrapper'ı ile tema tutarlılığını sağlayın
 */

// Örnek Composable fonksiyon
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExampleScreen(
    onButtonClick: () -> Unit = {},
    onNavigateBack: () -> Unit = {}
) {
    var counter by remember { mutableStateOf(0) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Top App Bar
        TopAppBar(
            title = { Text("Example Screen") },
            navigationIcon = {
                IconButton(onClick = onNavigateBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                }
            }
        )

        Spacer(modifier = Modifier.weight(1f))

        // Content
        Text(
            text = "Counter: $counter",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold
        )

        Button(
            onClick = {
                counter++
                onButtonClick()
            }
        ) {
            Text("Increment")
        }

        Spacer(modifier = Modifier.weight(1f))
    }
}

@Composable
private fun ExampleCard(
    title: String,
    subtitle: String,
    isSelected: Boolean = false
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected)
                MaterialTheme.colorScheme.primaryContainer
            else
                MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = title,
                fontWeight = FontWeight.SemiBold,
                fontSize = 16.sp
            )
            Text(
                text = subtitle,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 14.sp
            )
        }
    }
}

// MARK: - Preview Functions (Swift benzeri)

/**
 * Ana preview - varsayılan durum
 */
@Preview(showBackground = true)
@Composable
fun ExampleScreenPreview() {
    VisionVetAITheme {
        ExampleScreen()
    }
}

/**
 * Karanlık tema preview
 */
@Preview(showBackground = true)
@Composable
fun ExampleScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        ExampleScreen()
    }
}

/**
 * Küçük ekran boyutu preview
 */
@Preview(showBackground = true, widthDp = 320, heightDp = 568)
@Composable
fun ExampleScreenSmallDevicePreview() {
    VisionVetAITheme {
        ExampleScreen()
    }
}

/**
 * Büyük font boyutu preview - erişilebilirlik testi
 */
@Preview(showBackground = true, fontScale = 1.5f)
@Composable
fun ExampleScreenLargeFontPreview() {
    VisionVetAITheme {
        ExampleScreen()
    }
}

/**
 * Tablet boyutu preview
 */
@Preview(showBackground = true, widthDp = 840, heightDp = 480)
@Composable
fun ExampleScreenTabletPreview() {
    VisionVetAITheme {
        ExampleScreen()
    }
}

/**
 * Tek component preview'ı
 */
@Preview(showBackground = true)
@Composable
fun ExampleCardPreview() {
    VisionVetAITheme {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            ExampleCard(
                title = "Normal Card",
                subtitle = "This is a normal card"
            )
            ExampleCard(
                title = "Selected Card",
                subtitle = "This is a selected card",
                isSelected = true
            )
        }
    }
}

/**
 * Farklı durumlar için preview
 */
@Preview(showBackground = true)
@Composable
fun ExampleCardStatesPreview() {
    VisionVetAITheme {
        LazyColumn(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(5) { index ->
                ExampleCard(
                    title = "Item ${index + 1}",
                    subtitle = "Subtitle for item ${index + 1}",
                    isSelected = index == 2 // Only item 3 will be selected
                )
            }
        }
    }
}

/*
 * YENİ EKRAN OLUŞTURURKEN KULLANILACAK TEMPLATE:
 *
 * 1. Bu dosyayı kopyalayın
 * 2. Package adını değiştirin
 * 3. ExampleScreen'i kendi screen'inizle değiştirin
 * 4. Aşağıdaki preview türlerini ekleyin:
 *    - Normal preview
 *    - Dark theme preview
 *    - Small device preview
 *    - Large font preview
 *    - Component-specific preview'lar
 *
 * 5. Her preview fonksiyonunu Swift'teki gibi isimlendirin:
 *    - YourScreenPreview()
 *    - YourScreenDarkPreview()
 *    - YourComponentPreview()
 *
 * 6. VisionVetAITheme {} wrapper'ını her zaman kullanın
 *
 * 7. Preview parametreleri:
 *    - showBackground = true (beyaz arka plan)
 *    - widthDp, heightDp (cihaz boyutları)
 *    - fontScale (font boyutu)
 *    - device (cihaz türü)
 */
