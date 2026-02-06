import { configureStore } from "@reduxjs/toolkit";
import authReducer from "./slices/authSlice";
import patientReducer from "./slices/patientSlice";
import testSessionReducer from "./slices/testSessionSlice";
import aiAnalysisReducer from "./slices/aiAnalysisSlice";

export const store = configureStore({
  reducer: {
    auth: authReducer,
    patient: patientReducer,
    testSession: testSessionReducer,
    aiAnalysis: aiAnalysisReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ["auth/setUser"],
        ignoredPaths: ["auth.user"],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
