import { useState } from "react";

const formatNumber = (value) => {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    if (!file) {
      setError("먼저 엑셀 파일을 업로드해주세요.");
      return;
    }

    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "분석 중 오류가 발생했습니다.");
      }

      const payload = await response.json();
      setData(payload);
    } catch (submissionError) {
      setError(submissionError.message);
    } finally {
      setLoading(false);
    }
  };

  const renderResultsTable = () => {
    if (!data || !data.results || data.results.length === 0) {
      return <p>표시할 결과가 없습니다.</p>;
    }

    const columns = data.columns || Object.keys(data.results[0] || {});

    return (
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.results.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => (
                  <td key={`${rowIndex}-${column}`}>{formatNumber(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderAnalysis = () => {
    if (!data?.analysis) return null;
    const { analysis } = data;
    return (
      <section className="analysis">
        <h2>분석 요약</h2>
        <div className="metrics">
          <div className="metric">
            <span className="label">총 결과 수</span>
            <span className="value">{formatNumber(analysis.total_results)}</span>
          </div>
          <div className="metric">
            <span className="label">평균 리드 점수</span>
            <span className="value">
              {analysis.average_score !== null && analysis.average_score !== undefined
                ? analysis.average_score.toFixed(1)
                : "-"}
            </span>
          </div>
          <div className="metric">
            <span className="label">키워드 개수</span>
            <span className="value">{formatNumber(analysis.prompt_tokens?.length)}</span>
          </div>
        </div>

        <div className="grid">
          <div>
            <h3>리드 등급 분포</h3>
            <ul>
              {Object.entries(analysis.grade_counts || {}).map(([grade, count]) => (
                <li key={grade}>
                  {grade}: {formatNumber(count)}건
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3>감성 분포</h3>
            <ul>
              {Object.entries(analysis.sentiment_counts || {}).map(([sentiment, count]) => (
                <li key={sentiment}>
                  {sentiment}: {formatNumber(count)}건
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3>상위 추천 태그</h3>
            <ul>
              {(analysis.top_tags || []).map(([tag, count]) => (
                <li key={tag}>
                  {tag}: {formatNumber(count)}건
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>
    );
  };

  const renderStrategy = () => {
    if (!data?.ai_strategy) return null;
    return (
      <section className="strategy">
        <h2>AI 영업 전략 제안</h2>
        <div className="strategy-content">
          {data.ai_strategy.split("\n").map((line, index) => (
            <p key={`strategy-${index}`}>{line}</p>
          ))}
        </div>
      </section>
    );
  };

  return (
    <div className="container">
      <header>
        <h1>거래처 탐색 도우미</h1>
        <p>
          엑셀 거래처 데이터와 프롬프트를 업로드하면 규칙 기반 분석과 AI 전략을 제공하는
          웹 애플리케이션입니다.
        </p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="form">
          <label htmlFor="prompt">검색 프롬프트</label>
          <input
            id="prompt"
            name="prompt"
            type="text"
            placeholder="예: 서울 지역 병원 + 임플란트 관심"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
          />

          <label htmlFor="file">엑셀 파일 업로드</label>
          <input
            id="file"
            name="file"
            type="file"
            accept=".xlsx,.xls"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />

          <button type="submit" disabled={loading}>
            {loading ? "분석 중..." : "분석 시작"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {renderAnalysis()}
        {renderStrategy()}

        <section>
          <h2>검색 결과</h2>
          {renderResultsTable()}
        </section>
      </main>

      <footer>
        <p>© {new Date().getFullYear()} 거래처 탐색 도우미</p>
      </footer>

      <style jsx>{`
        .container {
          max-width: 1080px;
          margin: 0 auto;
          padding: 2rem 1.5rem 4rem;
          font-family: "Pretendard", "Noto Sans KR", sans-serif;
          color: #1f2933;
        }

        header {
          text-align: center;
          margin-bottom: 2.5rem;
        }

        header h1 {
          font-size: 2rem;
          margin-bottom: 0.5rem;
        }

        header p {
          color: #52606d;
        }

        .form {
          display: grid;
          gap: 1rem;
          padding: 1.5rem;
          border: 1px solid #d9e2ec;
          border-radius: 12px;
          background: #f8f9fb;
          margin-bottom: 2rem;
        }

        .form label {
          font-weight: 600;
        }

        .form input[type="text"],
        .form input[type="file"] {
          padding: 0.75rem 1rem;
          border-radius: 8px;
          border: 1px solid #cbd5e0;
          width: 100%;
        }

        .form button {
          margin-top: 0.5rem;
          padding: 0.9rem 1.2rem;
          border-radius: 10px;
          border: none;
          background: linear-gradient(135deg, #2563eb, #3b82f6);
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .form button:disabled {
          background: #94a3b8;
          cursor: not-allowed;
        }

        .form button:not(:disabled):hover {
          transform: translateY(-1px);
          box-shadow: 0 8px 16px rgba(37, 99, 235, 0.18);
        }

        .error {
          color: #e11d48;
          margin-bottom: 1rem;
        }

        section {
          margin-top: 2.5rem;
        }

        section h2 {
          font-size: 1.4rem;
          margin-bottom: 1rem;
        }

        .analysis .metrics {
          display: grid;
          gap: 1rem;
          grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
          margin-bottom: 1.5rem;
        }

        .metric {
          padding: 1rem;
          border-radius: 12px;
          border: 1px solid #e3e8ee;
          background: white;
          text-align: center;
        }

        .metric .label {
          display: block;
          margin-bottom: 0.4rem;
          color: #52606d;
        }

        .metric .value {
          font-size: 1.3rem;
          font-weight: 700;
          color: #1d4ed8;
        }

        .analysis .grid {
          display: grid;
          gap: 1.5rem;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .analysis ul {
          margin: 0;
          padding-left: 1.1rem;
        }

        .analysis li {
          margin-bottom: 0.25rem;
        }

        .strategy {
          border: 1px solid #d9e2ec;
          border-radius: 12px;
          padding: 1.5rem;
          background: #f1f5f9;
        }

        .strategy-content p {
          margin: 0.4rem 0;
          white-space: pre-wrap;
        }

        .table-wrapper {
          overflow-x: auto;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
        }

        table {
          width: 100%;
          border-collapse: collapse;
        }

        th,
        td {
          padding: 0.75rem;
          border-bottom: 1px solid #e2e8f0;
          text-align: left;
          font-size: 0.95rem;
          vertical-align: top;
        }

        th {
          background: #f1f5f9;
          font-weight: 700;
        }

        footer {
          margin-top: 3rem;
          text-align: center;
          color: #94a3b8;
          font-size: 0.9rem;
        }

        @media (max-width: 640px) {
          header h1 {
            font-size: 1.7rem;
          }

          .form {
            padding: 1.2rem;
          }
        }
      `}</style>
    </div>
  );
}

