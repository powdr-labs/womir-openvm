// CRUSH ExecutionState includes frame pointer (fp) between pc and timestamp.
#pragma once

template <typename T> struct CrushExecutionState {
    T pc;
    T fp;
    T timestamp;
};
