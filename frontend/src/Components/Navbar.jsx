import { useNavigate, useLocation } from "react-router-dom";
import { FaChartPie } from "react-icons/fa6";

const links = [
  { label: "Home", path: "/" },
  { label: "Predictions", path: "/prediction" },
  { label: "Performance", path: "/performance" },
  { label: "Backtest", path: "/backtest" },
];

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <div className="fixed top-0 left-0 z-50 flex h-16 w-full items-center justify-between bg-customBlack-100/95 px-6 text-white shadow-lg backdrop-blur">
      <div
        className="flex cursor-pointer items-center gap-2 text-2xl font-extrabold tracking-tight transition-colors hover:text-customGreen-100"
        onClick={() => navigate("/")}
      >
        <FaChartPie className="text-customGreen-100" />
        Investify
      </div>

      <div className="flex items-center gap-1">
        {links.map(({ label, path }) => {
          const active = location.pathname === path;
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200 ${
                active
                  ? "bg-customGreen-100 text-white shadow"
                  : "text-gray-300 hover:bg-white/10 hover:text-white"
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default Navbar;
