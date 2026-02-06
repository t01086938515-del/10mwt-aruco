"use client";

import { useState } from "react";
import { useAppDispatch } from "@/store/hooks";
import { addPatient } from "@/store/slices/patientSlice";
import { Modal, ModalFooter } from "@/components/ui/modal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";

interface AddPatientModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AddPatientModal({ isOpen, onClose }: AddPatientModalProps) {
  const dispatch = useAppDispatch();
  const [isLoading, setIsLoading] = useState(false);

  const [formData, setFormData] = useState({
    name: "",
    birth: "",
    gender: "male" as "male" | "female",
    diagnosis: "",
    height: "",
    weight: "",
    legLength: "",
    affectedSide: "none" as "left" | "right" | "both" | "none",
    assistiveDevice: "",
    afoUse: false,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const newPatient = {
        id: `p${Date.now()}`,
        name: formData.name,
        birth: formData.birth,
        gender: formData.gender,
        diagnosis: formData.diagnosis,
        height: Number(formData.height),
        weight: Number(formData.weight),
        legLength: formData.legLength ? Number(formData.legLength) : undefined,
        affectedSide: formData.affectedSide,
        assistiveDevice: formData.assistiveDevice || undefined,
        afoUse: formData.afoUse,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        testCount: 0,
      };

      dispatch(addPatient(newPatient));
      onClose();

      // Reset form
      setFormData({
        name: "",
        birth: "",
        gender: "male",
        diagnosis: "",
        height: "",
        weight: "",
        legLength: "",
        affectedSide: "none",
        assistiveDevice: "",
        afoUse: false,
      });
    } catch (error) {
      console.error("Failed to add patient:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="새 환자 등록"
      description="환자 정보를 입력해주세요."
      className="max-w-2xl"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          {/* Name */}
          <div className="space-y-2">
            <label className="text-sm font-medium">이름 *</label>
            <Input
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="홍길동"
              required
            />
          </div>

          {/* Birth */}
          <div className="space-y-2">
            <label className="text-sm font-medium">생년월일 *</label>
            <Input
              type="date"
              value={formData.birth}
              onChange={(e) => setFormData({ ...formData, birth: e.target.value })}
              required
            />
          </div>

          {/* Gender */}
          <div className="space-y-2">
            <label className="text-sm font-medium">성별 *</label>
            <Select
              value={formData.gender}
              onChange={(e) =>
                setFormData({ ...formData, gender: e.target.value as "male" | "female" })
              }
              options={[
                { value: "male", label: "남성" },
                { value: "female", label: "여성" },
              ]}
            />
          </div>

          {/* Diagnosis */}
          <div className="space-y-2">
            <label className="text-sm font-medium">진단명 *</label>
            <Input
              value={formData.diagnosis}
              onChange={(e) => setFormData({ ...formData, diagnosis: e.target.value })}
              placeholder="뇌졸중, 파킨슨병 등"
              required
            />
          </div>

          {/* Height */}
          <div className="space-y-2">
            <label className="text-sm font-medium">신장 (cm) *</label>
            <Input
              type="number"
              value={formData.height}
              onChange={(e) => setFormData({ ...formData, height: e.target.value })}
              placeholder="170"
              required
            />
          </div>

          {/* Weight */}
          <div className="space-y-2">
            <label className="text-sm font-medium">체중 (kg) *</label>
            <Input
              type="number"
              value={formData.weight}
              onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
              placeholder="65"
              required
            />
          </div>

          {/* Leg Length */}
          <div className="space-y-2">
            <label className="text-sm font-medium">하지 길이 (cm)</label>
            <Input
              type="number"
              value={formData.legLength}
              onChange={(e) => setFormData({ ...formData, legLength: e.target.value })}
              placeholder="85"
            />
          </div>

          {/* Affected Side */}
          <div className="space-y-2">
            <label className="text-sm font-medium">마비측</label>
            <Select
              value={formData.affectedSide}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  affectedSide: e.target.value as "left" | "right" | "both" | "none",
                })
              }
              options={[
                { value: "none", label: "없음" },
                { value: "left", label: "좌측" },
                { value: "right", label: "우측" },
                { value: "both", label: "양측" },
              ]}
            />
          </div>

          {/* Assistive Device */}
          <div className="space-y-2">
            <label className="text-sm font-medium">보조기구</label>
            <Input
              value={formData.assistiveDevice}
              onChange={(e) => setFormData({ ...formData, assistiveDevice: e.target.value })}
              placeholder="지팡이, 워커 등"
            />
          </div>

          {/* AFO Use */}
          <div className="flex items-center gap-2 self-end pb-2">
            <input
              type="checkbox"
              id="afoUse"
              checked={formData.afoUse}
              onChange={(e) => setFormData({ ...formData, afoUse: e.target.checked })}
              className="h-4 w-4 rounded border-[hsl(var(--border))]"
            />
            <label htmlFor="afoUse" className="text-sm font-medium">
              AFO 착용
            </label>
          </div>
        </div>

        <ModalFooter>
          <Button type="button" variant="outline" onClick={onClose} disabled={isLoading}>
            취소
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? "등록 중..." : "등록"}
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
