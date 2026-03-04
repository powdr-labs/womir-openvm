// WOMIR ExecutionState includes frame pointer (fp) between pc and timestamp.
#pragma once

template <typename T> struct WomirExecutionState {
    T pc;
    T fp;
    T timestamp;
};
