package com.visionvet.ai.core.database.repository

import com.visionvet.ai.core.database.dao.TransactionDao
import com.visionvet.ai.core.database.model.Transaction
import kotlinx.coroutines.flow.Flow

class TransactionRepository(
    private val transactionDao: TransactionDao
) {
    fun getAllTransactions(): Flow<List<Transaction>> = transactionDao.getAllTransactions()

    fun getPendingTransactions(): Flow<List<Transaction>> = transactionDao.getPendingTransactions()

    fun getRecentTransactions(hours: Int = 24): Flow<List<Transaction>> {
        val cutoffTime = System.currentTimeMillis() - (hours * 60 * 60 * 1000)
        return transactionDao.getRecentTransactions(cutoffTime)
    }

    suspend fun insertTransaction(transaction: Transaction) {
        transactionDao.insertTransaction(transaction)
    }

    suspend fun updateTransaction(transaction: Transaction) {
        transactionDao.updateTransaction(transaction)
    }

    suspend fun deleteTransaction(transaction: Transaction) {
        transactionDao.deleteTransaction(transaction)
    }
}
