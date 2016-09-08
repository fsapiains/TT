# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_DISTDIR=dist
CND_BUILDDIR=build


# Object Directory
OBJECTDIR=${CND_BUILDDIR}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/utils.o \
	${OBJECTDIR}/PrimalSVM.o \
	${OBJECTDIR}/main.o
#	${OBJECTDIR}/db.o \
	


# C Compiler Flags
CFLAGS=-Wall -g3 $(shell pkg-config --cflags opencv)

# CC Compiler Flags
CCFLAGS=-Wall -g3 $(shell pkg-config --cflags opencv)
CXXFLAGS=-Wall -g3 $(shell pkg-config --cflags opencv)

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=$(shell pkg-config --libs opencv) -lboost_filesystem -lboost_system -lm

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f Makefile ${CND_DISTDIR}/app

${CND_DISTDIR}/app: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}
	${LINK.cc} -o ${CND_DISTDIR}/app ${OBJECTFILES} ${LDLIBSOPTIONS} 
	${RM} -fr ${CND_BUILDDIR}
	${RM} -fr core


${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(CCC) $(CFLAGS) -c -o ${OBJECTDIR}/main.o main.cpp
	
	
${OBJECTDIR}/PrimalSVM.o: PrimalSVM.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(CCC) $(CFLAGS) -c -o ${OBJECTDIR}/PrimalSVM.o PrimalSVM.cpp

${OBJECTDIR}/utils.o: utils.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(CCC) $(CFLAGS) -c -o ${OBJECTDIR}/utils.o utils.cpp

#${OBJECTDIR}/tareas.o: tareas.c
#	${MKDIR} -p ${OBJECTDIR}
#	${RM} $@.d
#	$(CCC) $(CFLAGS) -c -o ${OBJECTDIR}/tareas.o tareas.c	

# Clean Targets
clean: ${CLEAN_SUBPROJECTS}
	${RM} -fr ${CND_BUILDDIR}
	${RM} -fr ${CND_DISTDIR}
	${RM} -fr core

# Subprojects
.clean-subprojects:
