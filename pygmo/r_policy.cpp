// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "handle_thread_py_exception.hpp"
#include "object_serialization.hpp"
#include "r_policy.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

r_pol_inner<py::object>::r_pol_inner(const py::object &o)
{
    // Forbid the use of a pygmo.r_policy as a UDRP.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::r_policy as a UDRP is forbidden and prevented by the fact
    // that the generic constructor from UDRP is disabled if the input
    // object is a pagmo::r_policy (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is an r_policy, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input r_policy.
    if (pygmo::type(o).is(py::module::import("pygmo").attr("r_policy"))) {
        pygmo::py_throw(PyExc_TypeError,
                        ("a pygmo.r_policy cannot be used as a UDRP for another pygmo.r_policy (if you need to copy a "
                         "replacement policy please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "r_policy");
    check_mandatory_method(o, "replace", "r_policy");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<r_pol_inner_base> r_pol_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return detail::make_unique<r_pol_inner>(m_value);
}

individuals_group_t
r_pol_inner<py::object>::replace(const individuals_group_t &inds, const vector_double::size_type &nx,
                                 const vector_double::size_type &nix, const vector_double::size_type &nobj,
                                 const vector_double::size_type &nec, const vector_double::size_type &nic,
                                 const vector_double &tol, const individuals_group_t &mig) const
{
    // NOTE: replace() may be called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string r_pol_name;
    try {
        r_pol_name = get_name();
    } catch (const py::error_already_set &) {
        pygmo::handle_thread_py_exception("Could not fetch the name of a pythonic replacement policy. The error is:\n");
    }

    try {
        // Fetch the new individuals in Python form.
        auto o = m_value.attr("replace")(pygmo::inds_to_tuple(inds), nx, nix, nobj, nec, nic,
                                         pygmo::vector_to_ndarr<py::array_t<double>>(tol), pygmo::inds_to_tuple(mig));

        // Convert back to C++ form and return.
        return pygmo::iterable_to_inds(o);
    } catch (const py::error_already_set &) {
        pygmo::handle_thread_py_exception("The replace() method of a pythonic replacement policy of type '" + r_pol_name
                                          + "' raised an error:\n");
    }
}

std::string r_pol_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string r_pol_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

template <typename Archive>
void r_pol_inner<py::object>::save(Archive &ar, unsigned) const
{
    ar << boost::serialization::base_object<r_pol_inner_base>(*this);
    ar << pygmo::object_to_vchar(m_value);
}

template <typename Archive>
void r_pol_inner<py::object>::load(Archive &ar, unsigned)
{
    ar >> boost::serialization::base_object<r_pol_inner_base>(*this);
    std::vector<char> v;
    ar >> v;
    m_value = pygmo::vchar_to_object(v);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_R_POLICY_IMPLEMENT(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

// Serialization support for the r_policy class.
py::tuple r_policy_pickle_getstate(const pagmo::r_policy &r)
{
    // The idea here is that first we extract a char array
    // into which r has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << r;
    }
    auto s = oss.str();
    return py::make_tuple(py::bytes(s.data(), boost::numeric_cast<py::size_t>(s.size())));
}

pagmo::r_policy r_policy_pickle_setstate(py::tuple state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialized the object.
    if (py::len(state) != 1) {
        pygmo::py_throw(PyExc_ValueError, ("the state tuple passed for replacement policy deserialization "
                                           "must have 1 element, but instead it has "
                                           + std::to_string(py::len(state)) + " element(s)")
                                              .c_str());
    }

    auto ptr = PyBytes_AsString(state[0].ptr());
    if (!ptr) {
        pygmo::py_throw(PyExc_TypeError, "a bytes object is needed to deserialize a replacement policy");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len(state[0])));
    pagmo::r_policy r;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> r;
    }

    return r;
}

} // namespace pygmo
