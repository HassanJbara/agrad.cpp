file(GLOB_RECURSE OBJECT_FILES "*.o" "*.a")
foreach(OBJECT_FILE ${OBJECT_FILES})
    file(REMOVE ${OBJECT_FILE})
endforeach()