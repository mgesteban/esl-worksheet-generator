"""FastAPI application for the ESL Worksheet Generator.

This module defines the FastAPI application and its endpoints for handling
student information, transcriptions, and worksheet generation.
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Session, select
from typing import List, Dict, Any

from app.db.database import get_session
from app.db.models import (
    Student, BasicInformation, ProfessionalBackground, PersonalBackground,
    InterestHobby, LearningContext, CulturalElements, SocialAspects,
    Interview, InterviewTemplate
)
from app.core.dependencies import get_student_extraction_service
from app.services.student_extractor import StudentExtractionService
from app.api.models.transcription import TranscriptionCreate
from app.api.models.student import StudentDetailResponse
from app.api.routes import worksheets, websocket

app = FastAPI(
    title="ESL Worksheet Generator API",
    description="API for generating personalized ESL worksheets based on student interviews",
    version="0.1.0"
)

# Include routers
app.include_router(worksheets.router, tags=["worksheets"])
app.include_router(websocket.router, tags=["websocket"])

@app.post("/transcriptions/", response_model=Dict[str, Any])
async def process_transcription(
    transcription_data: TranscriptionCreate,
    service: StudentExtractionService = Depends(get_student_extraction_service),
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """Process a transcription to extract student information.
    
    Args:
        transcription_data: Transcription data
        service: Student extraction service
        session: Database session
        
    Returns:
        Dict[str, Any]: Processed student information
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Extract information from transcription
        extracted_data = await service.process_transcription(transcription_data.transcription)
        
        # Save to database
        student = await service.save_student_info(session, extracted_data)
        
        return {
            "status": "success",
            "student_id": student.student_id,
            "extracted_info": extracted_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process transcription: {str(e)}"
        )

@app.get("/students/", response_model=List[StudentDetailResponse])
def get_students(
    skip: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session)
) -> List[StudentDetailResponse]:
    """Get a list of students with their basic information.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        session: Database session
        
    Returns:
        List[StudentDetailResponse]: List of student records with basic information
    """
    statement = select(Student).offset(skip).limit(limit)
    students = session.exec(statement).all()
    return students

@app.get("/students/{student_id}", response_model=StudentDetailResponse)
def get_student(
    student_id: int,
    session: Session = Depends(get_session)
) -> StudentDetailResponse:
    """Get detailed information about a student.
    
    Args:
        student_id: Student ID
        session: Database session
        
    Returns:
        StudentDetailResponse: Student record with basic information
        
    Raises:
        HTTPException: If student not found
    """
    # Query student with basic info
    statement = (
        select(Student)
        .where(Student.student_id == student_id)
    )
    student = session.exec(statement).first()
    
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
        
    return student

@app.get("/students/{student_id}/interviews", response_model=List[Interview])
def get_student_interviews(
    student_id: int,
    session: Session = Depends(get_session)
) -> List[Interview]:
    """Get all interviews for a student.
    
    Args:
        student_id: Student ID
        session: Database session
        
    Returns:
        List[Interview]: List of interview records
        
    Raises:
        HTTPException: If student not found
    """
    student = session.get(Student, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student.interviews
