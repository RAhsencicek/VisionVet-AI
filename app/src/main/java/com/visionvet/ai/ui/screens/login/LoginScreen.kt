package com.visionvet.ai.ui.screens.login

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.ui.theme.VisionVetAITheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LoginScreen(
    onLoginSuccess: () -> Unit,
    onNavigateToRegister: () -> Unit = {}
) {
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }
    var showErrorMessage by remember { mutableStateOf(false) }
    var showForgotPassword by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        Spacer(modifier = Modifier.weight(0.5f))



        // App title - Swift'teki .largeTitle.bold() gibi
        Text(
            text = "MİLCO",
            fontSize = 34.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )

        // Subtitle - Swift'teki .headline ve .secondary gibi
        Text(
            text = "Veterinary Parasite Analysis Tool",
            fontSize = 18.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // Login form - Swift'teki VStack(spacing: 16) gibi
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Username field - Swift'teki gray background ve rounded corners
            OutlinedTextField(
                value = username,
                onValueChange = { username = it },
                label = { Text("Email / Username") },
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Color.Gray.copy(alpha = 0.1f),
                        RoundedCornerShape(12.dp)
                    ),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color.Transparent,
                    unfocusedBorderColor = Color.Transparent,
                    focusedContainerColor = Color.Gray.copy(alpha = 0.1f),
                    unfocusedContainerColor = Color.Gray.copy(alpha = 0.1f)
                ),
                shape = RoundedCornerShape(12.dp)
            )

            // Password field - Swift'teki SecureField gibi
            OutlinedTextField(
                value = password,
                onValueChange = { password = it },
                label = { Text("Password") },
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        Color.Gray.copy(alpha = 0.1f),
                        RoundedCornerShape(12.dp)
                    ),
                visualTransformation = if (passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                singleLine = true,
                trailingIcon = {
                    IconButton(onClick = { passwordVisible = !passwordVisible }) {
                        Icon(
                            imageVector = if (passwordVisible) Icons.Filled.Lock else Icons.Filled.Person,
                            contentDescription = if (passwordVisible) "Hide password" else "Show password"
                        )
                    }
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color.Transparent,
                    unfocusedBorderColor = Color.Transparent,
                    focusedContainerColor = Color.Gray.copy(alpha = 0.1f),
                    unfocusedContainerColor = Color.Gray.copy(alpha = 0.1f)
                ),
                shape = RoundedCornerShape(12.dp)
            )

            // Error message - Swift'teki error handling gibi
            if (showErrorMessage) {
                Text(
                    text = "Invalid username or password",
                    color = MaterialTheme.colorScheme.error,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(top = 0.dp)
                )
            }
        }

        // Login button - Swift'teki primaryButtonStyle gibi
        Button(
            onClick = {
                isLoading = true
                // Simulate login process
                if (username.isNotEmpty() && password.isNotEmpty()) {
                    onLoginSuccess()
                } else {
                    showErrorMessage = true
                    isLoading = false
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            enabled = !isLoading && username.isNotEmpty() && password.isNotEmpty(),
            shape = RoundedCornerShape(12.dp)
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    color = Color.White,
                    modifier = Modifier.size(20.dp)
                )
            } else {
                Text(
                    text = "Login",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }

        // Forgot password - Swift'teki forgot password button gibi
        TextButton(
            onClick = { showForgotPassword = true },
            modifier = Modifier.padding(top = 8.dp)
        ) {
            Text(
                text = "Forgot Password?",
                color = MaterialTheme.colorScheme.primary,
                fontSize = 16.sp
            )
        }

        Spacer(modifier = Modifier.weight(1f))

        // Register option - Swift'teki HStack gibi
        Row(
            modifier = Modifier.padding(bottom = 20.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Don't have an account? ",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 16.sp
            )
            Text(
                text = "Sign Up",
                color = MaterialTheme.colorScheme.primary,
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.clickable { onNavigateToRegister() }
            )
        }
    }

    // Forgot password alert - Swift'teki Alert gibi
    if (showForgotPassword) {
        AlertDialog(
            onDismissRequest = { showForgotPassword = false },
            title = { Text("Forgot Password") },
            text = { Text("Please enter your email address to reset your password.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        showForgotPassword = false
                        // Handle password reset
                    }
                ) {
                    Text("Submit")
                }
            },
            dismissButton = {
                TextButton(onClick = { showForgotPassword = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun LoginScreenPreview() {
    VisionVetAITheme {
        LoginScreen(
            onLoginSuccess = {},
            onNavigateToRegister = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun LoginScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        LoginScreen(
            onLoginSuccess = {},
            onNavigateToRegister = {}
        )
    }
}

@Preview(showBackground = true, widthDp = 320, heightDp = 568)
@Composable
fun LoginScreenSmallDevicePreview() {
    VisionVetAITheme {
        LoginScreen(
            onLoginSuccess = {},
            onNavigateToRegister = {}
        )
    }
}

@Preview(showBackground = true, fontScale = 1.5f)
@Composable
fun LoginScreenLargeFontPreview() {
    VisionVetAITheme {
        LoginScreen(
            onLoginSuccess = {},
            onNavigateToRegister = {}
        )
    }
}
