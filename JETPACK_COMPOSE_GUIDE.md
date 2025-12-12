# Jetpack Compose Kullanım Kılavuzu

## 1. Temel Compose Yapısı

```kotlin
@Composable
fun MyScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Başlık",
            style = MaterialTheme.typography.headlineMedium
        )
        
        Button(
            onClick = { /* Tıklama işlemi */ }
        ) {
            Text("Butona Tıkla")
        }
    }
}
```

## 2. State Yönetimi

```kotlin
@Composable
fun CounterScreen() {
    var count by remember { mutableStateOf(0) }
    
    Column {
        Text("Sayı: $count")
        Button(
            onClick = { count++ }
        ) {
            Text("Artır")
        }
    }
}
```

## 3. LazyColumn/LazyRow (Liste Görünümleri)

```kotlin
@Composable
fun ItemList(items: List<String>) {
    LazyColumn {
        items(items) { item ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp)
            ) {
                Text(
                    text = item,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}
```

## 4. Navigation

```kotlin
@Composable
fun AppNavigation() {
    val navController = rememberNavController()
    
    NavHost(
        navController = navController,
        startDestination = "home"
    ) {
        composable("home") {
            HomeScreen(
                onNavigateToDetail = {
                    navController.navigate("detail")
                }
            )
        }
        composable("detail") {
            DetailScreen(
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}
```

## 5. Material Design 3 Bileşenleri

```kotlin
@Composable
fun MaterialComponents() {
    Column {
        // Card
        Card {
            Text("Card içeriği", modifier = Modifier.padding(16.dp))
        }
        
        // Floating Action Button
        FloatingActionButton(
            onClick = { /* işlem */ }
        ) {
            Icon(Icons.Default.Add, contentDescription = "Ekle")
        }
        
        // OutlinedTextField
        var text by remember { mutableStateOf("") }
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Metin girin") }
        )
        
        // Switch
        var checked by remember { mutableStateOf(false) }
        Switch(
            checked = checked,
            onCheckedChange = { checked = it }
        )
    }
}
```

## 6. Theming

```kotlin
@Composable
fun MyApp() {
    MaterialTheme(
        colorScheme = dynamicLightColorScheme(LocalContext.current),
        typography = Typography(),
        content = {
            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background
            ) {
                MyAppContent()
            }
        }
    )
}
```

## 7. Effect'ler

```kotlin
@Composable
fun DataScreen() {
    var data by remember { mutableStateOf<List<String>>(emptyList()) }
    
    // Ekran yüklendiğinde veri çek
    LaunchedEffect(Unit) {
        data = fetchDataFromNetwork()
    }
    
    // Lifecycle aware effect
    DisposableEffect(Unit) {
        val listener = createListener()
        onDispose {
            removeListener(listener)
        }
    }
}
```

## 8. Custom Composable'lar

```kotlin
@Composable
fun CustomButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true
) {
    Button(
        onClick = onClick,
        modifier = modifier,
        enabled = enabled,
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary
        )
    ) {
        Text(text)
    }
}

// Kullanım
CustomButton(
    text = "Özel Buton",
    onClick = { /* işlem */ },
    modifier = Modifier.fillMaxWidth()
)
```

## Projenizde Mevcut Compose Kullanımı

DashboardScreen.kt dosyanızda zaten şunları kullanıyorsunuz:
- `@Composable` fonksiyonlar
- `LazyColumn` ve `LazyVerticalGrid`
- `Card`, `Button`, `Text` bileşenleri
- `remember` ile state yönetimi
- Material Design 3 theming

Bu harika bir başlangıç! Daha fazla özellik eklemek isterseniz yukarıdaki örnekleri kullanabilirsiniz.
