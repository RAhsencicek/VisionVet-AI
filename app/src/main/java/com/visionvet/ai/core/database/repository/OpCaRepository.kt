package com.visionvet.ai.core.database.repository

import com.visionvet.ai.core.database.dao.UserDao
import com.visionvet.ai.core.database.dao.AnalysisDao
import com.visionvet.ai.core.database.model.User
import com.visionvet.ai.core.database.model.Analysis
import kotlinx.coroutines.flow.Flow

class OpCaRepository(
    private val userDao: UserDao,
    private val analysisDao: AnalysisDao
) {
    // User operations
    suspend fun getUserByUserId(userId: String): User? = userDao.getUserByUserId(userId)

    suspend fun getUserByEmail(email: String): User? = userDao.getUserByEmail(email)

    suspend fun insertUser(user: User) = userDao.insertUser(user)

    suspend fun updateUser(user: User) = userDao.updateUser(user)

    suspend fun updateLastLoginDate(userId: String, loginDate: Long) =
        userDao.updateLastLoginDate(userId, loginDate)

    // Analysis operations
    fun getAnalysesByUserId(userId: String): Flow<List<Analysis>> =
        analysisDao.getAnalysesByUserId(userId)

    suspend fun getAnalysisById(id: String): Analysis? = analysisDao.getAnalysisById(id)

    suspend fun getRecentAnalyses(userId: String, limit: Int = 10): List<Analysis> =
        analysisDao.getRecentAnalyses(userId, limit)

    suspend fun insertAnalysis(analysis: Analysis) = analysisDao.insertAnalysis(analysis)

    suspend fun updateAnalysis(analysis: Analysis) = analysisDao.updateAnalysis(analysis)

    suspend fun deleteAnalysis(analysis: Analysis) = analysisDao.deleteAnalysis(analysis)

    suspend fun getAnalysisCount(userId: String): Int =
        analysisDao.getAnalysisCountByUserId(userId)
}
