import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaChartPie, FaBars, FaXmark, FaSun, FaMoon } from "react-icons/fa6";
import { useTheme } from "../useTheme";

const links = [
  { label: "Home", path: "/" },
  { label: "Predictions", path: "/prediction" },
  { label: "Performance", path: "/performance" },
  { label: "Backtest", path: "/backtest" },
];

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);

  // Close the mobile menu whenever the route changes.
  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  // Don't leave the page scrollable behind the open mobile menu.
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  const go = (path) => {
    navigate(path);
    setOpen(false);
  };

  return (
    <header className="fixed top-0 left-0 z-50 w-full bg-customBlack-100/95 text-white shadow-lg backdrop-blur">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6">
        <button
          className="flex cursor-pointer items-center gap-2 text-xl font-extrabold tracking-tight transition-colors hover:text-customGreen-100 sm:text-2xl"
          onClick={() => go("/")}
        >
          <FaChartPie className="text-customGreen-100" />
          Investify
        </button>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-1 md:flex">
          {links.map(({ label, path }) => {
            const active = location.pathname === path;
            return (
              <button
                key={path}
                onClick={() => go(path)}
                aria-current={active ? "page" : undefined}
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
          <ThemeButton theme={theme} toggle={toggle} />
        </nav>

        {/* Mobile controls */}
        <div className="flex items-center gap-1 md:hidden">
          <ThemeButton theme={theme} toggle={toggle} />
          <button
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-menu"
            className="rounded-lg p-2.5 text-gray-200 transition hover:bg-white/10 hover:text-white"
          >
            {open ? <FaXmark size={20} /> : <FaBars size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <div
        id="mobile-menu"
        className={`overflow-hidden border-t border-white/10 transition-[max-height] duration-300 ease-out md:hidden ${
          open ? "max-h-80" : "max-h-0"
        }`}
      >
        <nav className="flex flex-col gap-1 bg-customBlack-100 px-4 py-3">
          {links.map(({ label, path }) => {
            const active = location.pathname === path;
            return (
              <button
                key={path}
                onClick={() => go(path)}
                aria-current={active ? "page" : undefined}
                className={`rounded-lg px-4 py-3 text-left text-base font-medium transition ${
                  active
                    ? "bg-customGreen-100 text-white"
                    : "text-gray-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                {label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

const ThemeButton = ({ theme, toggle }) => (
  <button
    onClick={toggle}
    aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
    title={theme === "dark" ? "Light mode" : "Dark mode"}
    className="ml-1 rounded-lg p-2.5 text-gray-200 transition hover:bg-white/10 hover:text-white"
  >
    {theme === "dark" ? <FaSun size={18} /> : <FaMoon size={18} />}
  </button>
);

export default Navbar;
