package com.visionvet.ai.di

import android.content.Context
import com.visionvet.ai.core.database.AppDatabase
import com.visionvet.ai.core.database.repository.TransactionRepository

object AppModule {
    private var database: AppDatabase? = null
    private var transactionRepository: TransactionRepository? = null

    fun provideDatabase(context: Context): AppDatabase {
        return database ?: synchronized(this) {
            database ?: AppDatabase.getDatabase(context).also { database = it }
        }
    }

    fun provideTransactionRepository(context: Context): TransactionRepository {
        return transactionRepository ?: synchronized(this) {
            transactionRepository ?: TransactionRepository(
                provideDatabase(context).transactionDao()
            ).also { transactionRepository = it }
        }
    }
}
