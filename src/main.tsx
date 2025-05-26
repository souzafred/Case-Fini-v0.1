import React from 'react';
import { createRoot } from 'react-dom/client';
import SalesIntelligenceDashboard from './sales_intelligence_dashboard';

const container = document.getElementById('root');
if (!container) throw new Error('Elemento #root não encontrado');

const root = createRoot(container);
root.render(
  <React.StrictMode>
    <SalesIntelligenceDashboard />
  </React.StrictMode>
);
