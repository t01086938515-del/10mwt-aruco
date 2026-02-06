import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface Patient {
  id: string;
  name: string;
  birth: string;
  gender: "male" | "female";
  diagnosis: string;
  height: number;
  weight: number;
  legLength?: number;
  affectedSide?: "left" | "right" | "both" | "none";
  assistiveDevice?: string;
  afoUse?: boolean;
  createdAt: string;
  updatedAt: string;
  lastTestDate?: string;
  testCount?: number;
}

interface PatientState {
  currentPatient: Patient | null;
  patientList: Patient[];
  loading: boolean;
  error: string | null;
  searchQuery: string;
}

const initialState: PatientState = {
  currentPatient: null,
  patientList: [],
  loading: false,
  error: null,
  searchQuery: "",
};

const patientSlice = createSlice({
  name: "patient",
  initialState,
  reducers: {
    setPatientList: (state, action: PayloadAction<Patient[]>) => {
      state.patientList = action.payload;
      state.loading = false;
    },
    setCurrentPatient: (state, action: PayloadAction<Patient | null>) => {
      state.currentPatient = action.payload;
    },
    addPatient: (state, action: PayloadAction<Patient>) => {
      state.patientList.unshift(action.payload);
    },
    updatePatient: (state, action: PayloadAction<Patient>) => {
      const index = state.patientList.findIndex(
        (p) => p.id === action.payload.id
      );
      if (index !== -1) {
        state.patientList[index] = action.payload;
      }
      if (state.currentPatient?.id === action.payload.id) {
        state.currentPatient = action.payload;
      }
    },
    deletePatient: (state, action: PayloadAction<string>) => {
      state.patientList = state.patientList.filter(
        (p) => p.id !== action.payload
      );
      if (state.currentPatient?.id === action.payload) {
        state.currentPatient = null;
      }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
      state.loading = false;
    },
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload;
    },
  },
});

export const {
  setPatientList,
  setCurrentPatient,
  addPatient,
  updatePatient,
  deletePatient,
  setLoading,
  setError,
  setSearchQuery,
} = patientSlice.actions;
export default patientSlice.reducer;
