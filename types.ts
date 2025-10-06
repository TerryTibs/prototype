// Fix: To resolve the "Cannot find namespace 'JSX'" error, React must be imported.
// The `JSX.Element` type requires the JSX namespace to be available.
import React from 'react';

export enum ModuleStatus {
  IMPLEMENTABLE_TODAY = "Implementable Today",
  ADVANCED_PROTOTYPE = "Advanced Prototype",
  RESEARCH_PROTOTYPE = "Research Prototype",
  CORE_INNOVATION = "Core Innovation",
}

export interface SraModule {
  name: string;
  acronym: string;
  description: string;
  existingTech: string;
  novelAspect: string;
  status: ModuleStatus;
  icon: JSX.Element;
}

export interface RoadmapStageData {
  stage: number;
  title: string;
  description: string;
  modules: SraModule[];
}
